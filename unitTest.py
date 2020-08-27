from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# added for preprocessing
import argparse
from data_process.util import create_dataset,unpadding,scale_raw, Params, set_logger,os_makedirs
from data_process.util import deepAR_dataset,deepAR_weight,deepAR_WeightedSampler

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit
# import torch
import logging

# added for training
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import models.deepAR as net


parser = argparse.ArgumentParser(
    description='PyTorch DeeepAR Time Series Forecasting')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-H', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-num_epochs', type=int, default=30, metavar='N',
                    help='epochs for training (default: 20)')

parser.add_argument('-k', type=int, default=5)
parser.add_argument('-sample-dense', action='store_true',default=True, help='Whether to continually sample the time series during preprocessing')
parser.add_argument('-restore', action='store_true', help='Whether to restore the model state from the best.pth.tar')

# parser.add_argument('--model-name', default='brent_model', help='Directory to save model state')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')


def pack_dataset(args):
    json_path = os.path.join('models', 'deepAR.params.json')
    params = Params(json_path)
    params.merge(args)

    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)

    dataset_params_path = 'data/paper/lag_settings.json'
    params.update(dataset_params_path)
    
    params.steps = params.datasets[params.dataset]['lag_order']

    params.train_window = params.steps+params.H
    params.test_window = params.train_window
    params.predict_start = params.steps

    params.predict_steps=params.H
    # params.num_class = 1
    params.cov_dim = 0

    # params.num_epochs = 30
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)
    tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = tscv.split(dataset)
    train_idx, test_idx = last
    train_data, test_data = dataset[train_idx],dataset[test_idx]

    train_tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = train_tscv.split(train_data)
    train_idx, valid_idx = last
    train_data, valid_data = dataset[train_idx],dataset[valid_idx]

    train_set, train_input = deepAR_dataset(train_data,train=True,h=params.H,steps=params.steps,sample_dense=args.sample_dense)
    valid_set, valid_input = deepAR_dataset(valid_data,train=True,h=params.H,steps=params.steps,sample_dense=args.sample_dense)
    test_set, test_input = deepAR_dataset(test_data,train=False,h=params.H,steps=params.steps,sample_dense=args.sample_dense)
    test_target = test_input[:,params.predict_start:,0]

    train_input, train_v = deepAR_weight(train_input,params.steps)
    train_sample = deepAR_WeightedSampler(train_v)

    params.batch_size = 128
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=train_sample, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=valid_set.samples,
                            sampler=RandomSampler(valid_set), num_workers=4)
    
    params.model_name = '{}_h{}_deepAR'.format(params.dataset,params.H)
    params.model_dir = os.path.join('experiments', params.model_name)
    params.plot_dir = os.path.join(params.model_dir, 'figures')
    # create missing directories
    os_makedirs(params.plot_dir)

    i=-1
    params.cv = i
    logger = logging.getLogger('deepAR.cv{}'.format(i))
    set_logger(os.path.join(params.model_dir, 'train.cv{}.log'.format(i)), logger)

    logger.info('Loading the datasets for batch-training')


    params.restore = False
    return params, logger, train_loader, valid_loader, test_input, test_target


if __name__ == "__main__":
    args = parser.parse_args()
    params, logger, train_loader, valid_loader, test_input, test_target = pack_dataset(args)

    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params, logger).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params,logger)    

    logger.info('Loading complete.')
    logger.info(f'Model: \n{str(model)}')

    model.xfit(train_loader,valid_loader)

    # print(test_input.shape)
    pred = model.point_predict(test_input)
    trmse = np.sqrt(mean_squared_error(test_target, pred))
    logger.info('{}\t H: {}\t Testing RMSE:{}'.format(
        params.dataset, params.H, trmse))
