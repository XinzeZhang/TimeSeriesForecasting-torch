import math
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import torch
import shutil
import logging
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


matplotlib.use('Agg')
#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions


# logger = logging.getLogger('DeepAR.Data')


def scale_raw(raw):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # raw = raw.reshape(raw.shape[0], 1)
    scaler = scaler.fit(raw)
    # transform train
    norm_raw = scaler.transform(raw)
    norm_raw = norm_raw[:, 0]
    return scaler, norm_raw


def toTorch(train_input, train_target, test_input, test_target):
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()
    return train_input, train_target, test_input, test_target


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset


def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] - 1)

    for i in range(s.shape[0]):
        s[i] = np.diagonal(np.flip(a, 1), offset=-i + h-1,
                           axis1=0, axis2=1).copy().mean()

    return s


class scaled_Dataset(Dataset):
    '''
    Packing the input x_data and label_data to torch.dataset
    '''

    def __init__(self, x_data, label_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.samples = self.data.shape[0]
        # logger.info(f'samples: {self.samples}')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (self.data[index, :, :], self.label[index])


class multiClass_Dataset(Dataset):
    '''
    only support multiple class
    '''

    def __init__(self, x_data, label_data, v_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.v = v_data.copy()
        self.test_len = self.data.shape[0]
        # logger.info(f'test_len: {self.test_len}')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index, :, :-1], int(self.data[index, 0, -1]), self.v[index], self.label[index])


def deepAR_dataset(data, train=True, h=None, steps=None, sample_dense=True):
    assert h != None and steps != None
    raw_data = unpadding(data).reshape(-1, 1)
    time_len = raw_data.shape[0]
    input_size = steps
    window_size = h + steps
    stride_size = h
    if not sample_dense:
        windows_per_series = np.full((1), (time_len-input_size) // stride_size)
    else:
        windows_per_series = np.full((1), 1 + time_len-window_size)
    total_windows = np.sum(windows_per_series)

    x_input = np.zeros((total_windows, window_size, 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')

    for i in range(windows_per_series[series]):
        # get the sample with minimal time period, in this case. which is 24 points (24h, 1 day)
        stride = 1
        if not sample_dense:
            stride = stride_size

        window_start = stride*i
        window_end = window_start+window_size
        '''
        print("x: ", x_input[count, 1:, 0].shape)
        print("window start: ", window_start)
        print("window end: ", window_end)
        print("data: ", data.shape)
        print("d: ", data[window_start:window_end-1, series].shape)
        '''
        # using the observed value in the t-1 step to forecast the t step, thus the first observed value in the input should be t0 step and is 0, as well as the first value in the labels should be t1 step.

        x_input[count, 1:, 0] = raw_data[window_start:window_end-1, series]
        label[count, :] = raw_data[window_start:window_end, series]
        # get the nonzero number of the input observed values

    packed_dataset = scaled_Dataset(x_data=x_input, label_data=label)
    return packed_dataset, x_input


class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def merge(self, args):
        new = vars(args)
        for key in new:
            if not key in self.dict:
                self.dict[key] = new[key]

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


def set_logger(log_path, logger):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    logger = logging.getLogger('DeepAR')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(TqdmHandler(fmt))


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(
            checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(
            f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')
    print('Checkpoint saved to {}'.format(filepath))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        # logger.info('Best checkpoint copied to best.pth.tar')
        print('Best checkpoint copied to best.pth.tar')


def savebest_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(checkpoint, 'best.pth.tar')
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.params.num_epochs -= checkpoint['epoch'] + 1

    return checkpoint


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics


def get_metrics(sample_mu, labels, predict_start, samples=None, relative=False):
    metric = dict()
    metric['ND'] = accuracy_ND_(
        sample_mu, labels[:, predict_start:], relative=relative)
    metric['RMSE'] = accuracy_RMSE_(
        sample_mu, labels[:, predict_start:], relative=relative)
    if samples is not None:
        metric['rou90'] = accuracy_ROU_(
            0.9, samples, labels[:, predict_start:], relative=relative)
        metric['rou50'] = accuracy_ROU_(
            0.5, samples, labels[:, predict_start:], relative=relative)
    return metric


def update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, predict_start, samples=None, relative=False):
    raw_metrics['ND'] = raw_metrics['ND'] + \
        accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + \
        accuracy_RMSE(sample_mu, labels[:, predict_start:], relative=relative)
    input_time_steps = input_mu.numel()
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [deep_ar_loss_fn(
        input_mu, input_sigma, labels[:, :predict_start]) * input_time_steps, input_time_steps]
    if samples is not None:
        raw_metrics['rou90'] = raw_metrics['rou90'] + \
            accuracy_ROU(
                0.9, samples, labels[:, predict_start:], relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + \
            accuracy_ROU(
                0.5, samples, labels[:, predict_start:], relative=relative)
    return raw_metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
        raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (
        raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / \
            raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / \
            raw_metrics['rou50'][1]
    return summary_metric


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(
            torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul(
        (mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            print('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(
                samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) +
                     (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                           predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                           alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})

        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def deep_ar_loss_fn(mu: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (torch.Tensor) dimension [batch_size] - estimated mean at time step t
        sigma: (torch.Tensor) dimension [batch_size] - estimated standard deviation at time step t
        labels: (torch.Tensor) dimension [batch_size] z_t
    Returns:
        loss: (torch.Tensor) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(
        mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)
