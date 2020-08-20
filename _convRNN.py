import os
import numpy as np
from tqdm import tqdm
# added for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import argparse
from data_process.util import create_dataset,unpadding, Params, set_logger
from data_process.util import scaled_Dataset, de_scale

import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler

from models.convRNN import ConvRNN
import logging

logger = logging.getLogger('ConvRNN')

parser = argparse.ArgumentParser(
    description='PyTorch DeeepAR Time Series Forecasting')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-H', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-steps', type=int, default=24)
parser.add_argument('-num_epochs', type=int, default=1000, metavar='N',
                    help='epochs for training (default: 1000)')
parser.add_argument('-sample-dense', action='store_true',default=True, help='Whether to continually sample the time series during preprocessing')
parser.add_argument('-restore', action='store_true', help='Whether to restore the model state from the best.pth.tar')
parser.add_argument('-k', type=int, default=5)
# parser.add_argument('--model-name', default='brent_model', help='Directory to save model state')
# parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
# parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')

def pack_dataset(ts, steps, H):
    _ = create_dataset(ts ,look_back= steps + H - 1)
    X, Y = _[:, :(0 - H)], _[:, (0-H):].reshape(-1, H)
    X = X[:,:, np.newaxis]
    return X, Y

if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join('models', 'convRNN.params.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = Params(json_path)

    params.merge(args)

    params.dataset = 'london_2013_summary'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    
    params.steps=168
    params.H=24
    params.cov_dim = 0
    # test
    params.num_epochs = 1000

    params.model_name = '{}_h{}_convRNN'.format(params.dataset,params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    # ts = unpadding(dataset)

    # # ts_train = ts[:-9*24].copy()
    # # ts_val = ts[-9*24-168:-7*24].copy()
    # # ts_test= ts[-7*24-168:].copy()
    kf = KFold(n_splits=params.k)
    kf.get_n_splits(dataset)

    params.restore = False
    cvs = []
    for i, (train_idx, test_idx) in tqdm(enumerate(kf.split(dataset))):
        params.cv = i
        train_data = dataset[train_idx]
        test_data = dataset[test_idx]
        x_train = train_data[:, :(
            0 - params.H)].reshape(train_data.shape[0], params.steps,1)
        y_train = train_data[:, (0-params.H):].reshape(-1, params.H)
        x_test = test_data[:, :(0 - params.H)
                           ].reshape(test_data.shape[0], params.steps,1)
        y_test = test_data[:, (0-params.H):].reshape(-1, params.H)


        train_set = scaled_Dataset(x_data=x_train, label_data=y_train)
        test_set = scaled_Dataset(x_data=x_test, label_data=y_test)

        params.batch_size = len(train_set) // 4
        train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
        val_loader = DataLoader(test_set, batch_size=test_set.samples,
                                sampler=RandomSampler(test_set), num_workers=4)
        logger.info('Loading complete.')

        model_dir = os.path.join('experiments', params.model_name)
        params.model_dir = model_dir
        params.plot_dir = os.path.join(model_dir, 'figures')
        # create missing directories
        try:
            os.makedirs(params.plot_dir)
        except FileExistsError:
            pass

        logger = logging.getLogger('convRNN.cv{}'.format(i))
        set_logger(os.path.join(model_dir, 'train.cv{}.log'.format(i)), logger)

        # use GPU if available
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda')
            # torch.cuda.manual_seed(240)
            logger.info('Using Cuda...')
            model = ConvRNN(params, logger).cuda()
        else:
            params.device = torch.device('cpu')
            # torch.manual_seed(230)
            logger.info('Not using cuda...')
            model = ConvRNN(params,logger)

        logger.info('Loading the datasets for batch-training')

        logger.info(f'Model: \n{str(model)}')


        model.xfit(train_loader,val_loader,restore_file=os.path.join(params.model_dir,'best.pth.tar'))

        print(x_test.shape)
        pred = model.predict(x_test)
        pred = de_scale(params, scaler, pred)
        target = de_scale(params, scaler, y_test)
        cvs.append((pred, target))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    measures = np.zeros(len(cvs))
    for i, (pred, target) in enumerate(cvs):
        vrmse = np.sqrt(mean_squared_error(target, pred))
        measures[i] = vrmse
        logger.info('{}\t H: {}\t CV: {}\t RMSE:{}'.format(
            params.dataset, params.H, i, vrmse))
        logger.info(params.dataset + '\t H: ' + str(params.H) +
              '\t Avg RMSE: ' + str(measures.mean()) + '\t')