import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import logging
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from data_process.metric import rmse, mape, smape
from data_process.util import Params, set_logger, os_rmdirs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import gc
from tqdm import trange
from data_process.util import os_makedirs
from data_process.util import plot_xfit
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.nn.init as Init
import torch.nn as nn
import torch
from data_process.dataset import create_dataset, de_scale, scaled_Dataset, difference, inverse_diff

# from memory_profiler import profile
# using python -m memory_profiler models/ESC_CNN.py to show the function memoery usage


def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    if model.params.device == torch.device('cpu'):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint, map_location='cuda')

    model.channels = checkpoint['channels']
    model.best_valid_channels = checkpoint['best_valid_channels']
    model.conv_weight = checkpoint['conv_weight']
    model.conv_bias = checkpoint['conv_bias']
    model.out_weight = checkpoint['out_weight']
    # model.loss_list = checkpoint['loss_list']
    # model.vloss_list = checkpoint['vloss_list']


def save_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(
        checkpoint, 'train.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)

# @profile


def cnn_dataset(data, h, steps):
    x = data[:, :(0 - h)].reshape(data.shape[0], 1, steps)
    y = data[:, (0-h):].reshape(-1, h)
    data_set = scaled_Dataset(x_data=x, label_data=y)

    return data_set, x, y


class Filter_Serach():
    def __init__(self, params=None, logger=None):
        super(Filter_Serach, self).__init__()
        self.params = params
        self.logger = logger
        self.In_channels = params.In_channels
        # self.Candidate_size = params.candidate_size
        self.Candidate_size = 1
        self.Kernel_size = params.kernel_size
        self.Kernel_list = params.kernel_list
        self.p_size = 3

        self.Lambdas = params.Lambdas
        self.Lambdas_std = params.Lambdas_std
        self.r = params.r
        self.tolerance = params.tolerance

        self.Pool = nn.AvgPool1d(kernel_size=self.p_size, stride=1, padding=0)

        self.loss_fn = nn.MSELoss()

        # mp.set_start_method('spawn')
        # self.share_memory()

    def solve_output(self, feature, target):
        with torch.no_grad():
            C_state = torch.cat(
                (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
            output_w, _ = torch.lstsq(target, C_state)
            output_w = output_w[0:C_state.size(1)].to(self.params.device)
        return output_w

    def conv_transform(self, input, conv_weight, conv_bias):
        with torch.no_grad():
            fm_list = []
            for cw,cb in zip(conv_weight,conv_bias):
                k_size = cw.data.size(2)
                Conv = nn.Conv1d(self.In_channels, 1, k_size,
                                padding=0).to(self.params.device)
                Conv.weight.data = cw
                Conv.bias.data = cb

                feature_map = Conv(input)
                feature_map = torch.sigmoid(feature_map)
                feature_map = self.Pool(feature_map)
                feature_map = feature_map.view(
                    feature_map.data.size(0), feature_map.data.size(2))
                fm_list.append(feature_map)
            fm = torch.cat(fm_list,1)
        return fm

    def random_search_lambda(self, input, error, conv_weight, conv_bias, Lambda):
        self.conv_weight, self.conv_bias, self.output_w = None, None, None

        min_loss = 9999
        for j, kernel_size in enumerate(self.Kernel_list):
            weight_candidate = torch.empty(
                self.Candidate_size, self.In_channels, kernel_size).uniform_(-Lambda, Lambda).float().to(self.params.device)
            bias_candidates = torch.empty(
                self.Candidate_size).uniform_(-Lambda, Lambda).float().to(self.params.device)
            
            temp_cw = conv_weight.copy()
            temp_cw.append(weight_candidate)

            temp_cb = conv_bias.copy()
            temp_cb.append(bias_candidates)

            fm = self.conv_transform(input,temp_cw,temp_cb)

            output_w = self.solve_output(fm,error)
            
            fm = torch.cat((torch.ones(fm.size(0),1).to(self.params.device),fm),1)
            
            pred = fm.mm(output_w)
            loss = self.loss_fn(pred, error).item()
            if loss < min_loss:
                min_loss = loss
                self.conv_weight = weight_candidate[0,:,:].reshape(
            1, self.In_channels, kernel_size).detach().clone()
                self.conv_bias = bias_candidates[0].view(
            -1).detach().clone()
                self.output_w = output_w.detach().clone()
                self.Kernel_size = kernel_size
            
        self.logger.info('Find channel with Kernel size: {} \t Lambda: {}: \t Best Score: {:.4f}'.format(
            self.Kernel_size, Lambda, min_loss))

        return True

    def random_search(self, input, error, conv_weight, conv_bias,):
        with torch.no_grad():
            find = False
            self.params.num_process = 8
            for Lambda in self.Lambdas:
                find = self.random_search_lambda(
                    input, error, conv_weight, conv_bias, Lambda)
                if find:
                    break

            return find, self.conv_weight, self.conv_bias, self.output_w


class SM_CNN(nn.Module):
    def __init__(self, params=None, logger=None):
        super(SM_CNN, self).__init__()
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.In_channels = params.In_channels
        self.Timesteps = params.steps
        self.Output_dim = params.H
        self.Channel_size = params.channel_size
        self.Candidate_size = params.candidate_size
        self.Kernel_size = params.kernel_size
        self.Kernel_list = params.kernel_list
        self.p_size = 3

        self.Lambdas = params.Lambdas
        self.Lambdas_std = params.Lambdas_std
        self.r = params.r
        self.tolerance = params.tolerance

        with torch.no_grad():
            self.init_convWeight = Init.uniform_(torch.empty(
                1, self.In_channels, self.Kernel_size).float(), a=-0.25, b=0.25).to(self.params.device)
            self.init_convBias = Init.uniform_(
                torch.empty(1).float(), a=-0.25, b=0.25).to(self.params.device)
        self.channels = 1
        self.best_valid_channels = 1

        self.conv_weight = []
        self.conv_weight.append(self.init_convWeight)
        self.conv_bias = []
        self.conv_bias.append(self.init_convBias)

        self.out_weight = []

        # self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,self.Kernel_size,padding=0)
        self.Pool = nn.AvgPool1d(kernel_size=self.p_size, stride=1, padding=0)
        # self.fc  = nn.Linear(self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2) ,self.Output_dim,bias=False)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []
        self.vloss_list = []

        self.params.plot_dir = os.path.join(params.model_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

    # @profile
    def forward(self, input, channels):
        fm = self.conv_transform(input, channels)
        pred = self.fc_transform(fm, channels)
        return pred

    def conv_transform(self, input, channels):
        fm_list = []
        for channel in range(channels):
            conv_fm = self.channel_transform(input,self.conv_weight[channel],self.conv_bias[channel])
            fm_list.append(conv_fm)
        fm = torch.cat(fm_list,1)
        return fm
    # @profile
    def channel_transform(self, input, conv_weight, conv_bias):
        '''
        Using list to storage the conv-weight for multiple kernel-size filters
        '''
        k_size = conv_weight.data.size(2)
        Conv = nn.Conv1d(self.In_channels, 1, k_size,
                         padding=0).to(self.params.device)
        Conv.weight.data = conv_weight
        Conv.bias.data = conv_bias

        feature_map = Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        feature_map = feature_map.view(
            feature_map.data.size(0), feature_map.data.size(2))

        # del Conv
        # gc.collect()
        return feature_map

    # @profile
    def solve_output(self, feature, target):
        with torch.no_grad():
            C_state = torch.cat(
                (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
            output_w, _ = torch.lstsq(target, C_state)
            output_w = output_w[0:C_state.size(1)].to(self.params.device)
        return output_w

    # @profile
    def fc_transform(self, feature, epoch):
        C_state = torch.cat(
            (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
        pred = C_state.mm(self.out_weight[epoch-1])

        # del C_state
        # gc.collect()
        return pred

    # @profile
    def xfit(self, train_loader, val_loader, restore_file=None):
        with torch.no_grad():
            min_vmse = 9999
            train_x, train_y = None, None
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()

            val_x, val_y = None, None
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                val_x, val_y = batch_x.detach().clone(), batch_y.detach().clone()

            if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
                self.logger.info(
                    'Restoring parameters from {}'.format(restore_file))
                load_checkpoint(restore_file, self)
            else:
                # calcaulate the initial fit error with the first channel
                C_state_X = self.channel_transform(
                    train_x, self.init_convWeight, self.init_convBias)
                output_w = self.solve_output(
                    C_state_X, train_y.detach().clone())
                self.out_weight.append(output_w)

            loss = 9999
            for i in trange(self.Channel_size):
                if i == self.channels:
                    filter_search = Filter_Serach(self.params, self.logger)
                    success, new_conv_weight, new_conv_bias, new_output_w = None, None, None, None
                    if self.params.search == 'random':
                        success, new_conv_weight, new_conv_bias, new_output_w = filter_search.random_search(
                            train_x, error, self.conv_weight, self.conv_bias)

                    assert success is not None
                    if not success or loss < self.tolerance:
                        break
                    else:
                        self.conv_weight.append(new_conv_weight)
                        self.conv_bias.append(new_conv_bias)
                        self.out_weight.append(new_output_w)
                        self.channels += 1

                # update error
                pred = self.forward(train_x, i+1)
                error = train_y - pred
                loss = self.loss_fn(pred, train_y).item()
                self.loss_list.append(loss)

                vpred = self.forward(val_x, i+1)
                vloss = self.loss_fn(vpred, val_y).item()
                self.vloss_list.append(vloss)

                # vmse = vloss
                # self.logger.info('Current vmse: {:.4f}'.format(vmse))
                if vloss < min_vmse:
                    min_vmse = vloss
                    self.best_valid_channels = self.channels
                    self.logger.info('Found new best state')
                    self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

                self.logger.info('Channel size: {} \t Kernel size: {} \nTraining MSE: {:.8f} \t Validating MSE: {:.8f} \t Best VMSE: {:.8f}'.format(
                    i+1, self.conv_weight[i].data.size(2), loss, vloss, min_vmse))

                save_checkpoint({
                    'channels': self.channels,
                    'best_valid_channels': self.best_valid_channels,
                    'cv': self.params.cv,
                    'conv_weight': self.conv_weight,
                    'conv_bias': self.conv_bias,
                    'out_weight': self.out_weight,
                    'loss_list': self.loss_list,
                    'vloss_list': self.vloss_list}, checkpoint=self.params.model_dir)

                self.logger.info(
                    'Checkpoint saved to {}'.format(self.params.model_dir))

                gc.collect()

            plot_xfit(np.array(self.loss_list), np.array(self.vloss_list),
                      self.params.dataset + '_cv{}_loss'.format(self.params.cv), self.params.plot_dir)

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.model_dir,
                                'train.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth) and using_best:
            self.logger.info(
                'Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self)
        with torch.no_grad():
            x = torch.tensor(x).to(torch.float32).to(self.params.device)
            output = self.forward(x, self.best_valid_channels)
            pred = output.detach().cpu().numpy()

        return pred


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    json_path = os.path.join('models', 'ESC_CNN.params.json')
    params = Params(json_path)
    params.dataset = 'ar1'
    params.H = 5
    params.k = 5

    params.test = True
    params.experiment = 'test'
    params.diff = True
    params.normal = True  # False
    params.search = 'random'
    if params.normal:
        params.experiment = os.path.join(params.experiment, 'normal')
        scaler = StandardScaler()
    else:
        params.experiment = os.path.join(params.experiment, 'minmax')
        scaler = MinMaxScaler(feature_range=(-1, 1))
    params.scaler = scaler

    if params.diff:
        params.experiment += '_diff'

    params.model_name = 'smCNN_{}_h{}_{}'.format(
        params.dataset, params.H, params.search)
    params.model_dir = os.path.join(params.experiment, params.model_name)
    params.candidate_size = 20
    params.channel_size = 10
    params.Lambdas = [1, 5, 15, 30, 50, 100, 150, 200]
    # params.r = [0.999999, 0.9, 0.7, 0.5]
    params.r = [0.8, 0.9, 0.99, 0.9999, 0.999999]
    # os_rmdirs(params.model_dir)

    ts = np.load('./data/paper.esm/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    df_ts = difference(ts)

    dataset_params_path = 'data/paper.esm/lag_settings.json'
    params.update(dataset_params_path)

    params.steps = params.datasets[params.dataset]['lag_order']

    # params.num_class = 1
    params.cov_dim = 0

    raw_dataset = create_dataset(ts, look_back=params.steps + params.H - 1)
    if params.diff:
        dataset = create_dataset(df_ts, look_back=params.steps + params.H - 1)
    else:
        dataset = raw_dataset

    tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = tscv.split(dataset)
    train_idx, test_idx = last
    raw_test_data = raw_dataset[test_idx]

    dataset = scaler.fit_transform(dataset)
    train_data, test_data = dataset[train_idx], dataset[test_idx]

    train_tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = train_tscv.split(train_data)
    train_idx, valid_idx = last
    train_data, valid_data = dataset[train_idx], dataset[valid_idx]
    if params.diff:
        train_data = train_data[1:, :]

    train_set, _, _ = cnn_dataset(
        train_data, h=params.H, steps=params.steps)
    valid_set, _, _ = cnn_dataset(
        valid_data, h=params.H, steps=params.steps)
    test_set, test_input, test_target = cnn_dataset(
        test_data, h=params.H, steps=params.steps)

    train_loader = DataLoader(train_set, batch_size=train_set.samples)
    valid_loader = DataLoader(valid_set, batch_size=valid_set.samples)

    params.In_channels = 1 + params.cov_dim

    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')

    # params.device = torch.device('cpu')

    params.restore = True

    params.kernel_size = int(params.steps // 4)
    params.kernel_list = [int(params.steps / 3), int(params.steps / 4), int(params.steps / 5), int(params.steps / 6)]

    times = 1
    measures = np.zeros((times, 3))
    for i in range(times):
        if os.path.exists(os.path.join(params.model_dir, 'measures.npy')):
            measures = np.load(os.path.join(
                params.model_dir, 'measures.npy'))
            # if measures[i, 0].item() != 0.0:
            #     continue

        params.cv = i
        logger = logging.getLogger('escCNN.cv{}'.format(i))
        set_logger(os.path.join(params.model_dir,
                                'train.cv{}.log'.format(i)), logger)
        logger.info(
            'Loading the datasets and model for {}th-batch-training'.format(i))
        # use GPU if available
        model = SM_CNN(params, logger)
        logger.info('Loading complete.')
        logger.info(f'Model: \n{str(model)}')
        if not os.path.exists(os.path.join(params.model_dir, 'train.cv{}.pth.tar'.format(i+1))):
            model.xfit(train_loader, valid_loader, restore_file=os.path.join(
                params.model_dir, 'train.cv{}.pth.tar'.format(i)))

        pred = model.predict(test_input)

        del model
        gc.collect()

        if params.diff:
            _test_target, _pred = inverse_diff(params, pred, raw_test_data)
        else:
            _test_target, _pred = de_scale(
                params, test_target), de_scale(params, pred)

        trmse = rmse(_test_target, _pred)
        tmape = mape(_test_target, _pred)
        tsmape = smape(_test_target, _pred)
        logger.info('{}\t H: {}\t Trail: {} \nTesting RMSE: {:.4g} \t MAPE: {:.4g} \t SMAPE: {:.4g}'.format(
            params.dataset, params.H, i, trmse, tmape, tsmape))
        measures[i, 0] = trmse
        measures[i, 1] = tmape
        measures[i, 2] = tsmape

        np.save(os.path.join(params.model_dir, 'measures'), measures)
