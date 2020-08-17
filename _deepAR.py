from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# added for preprocessing
import argparse
from data_process.util import create_dataset,unpadding,scale_raw, prep_data, Params, set_logger

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# import torch
import logging

# added for training
import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler

import model.deepAR as net

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logger = logging.getLogger('DeepAR.Train')

parser = argparse.ArgumentParser(
    description='PyTorch DeeepAR Time Series Forecasting')
parser.add_argument('-step', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-dim', type=int, default=24)
parser.add_argument('-sample-dense', action='store_true',default=True, help='Whether to sample during evaluation')

parser.add_argument('--model-name', default='brent_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'



if __name__ == "__main__":
    args = parser.parse_args()

    data = args.dataset
    data ='MT_001'
    ts = np.load('./data/paper/{}.npy'.format(data))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    segmentation = int(len(ts)*5/6)
    # ts = ts.reshape(-1,1)
    # scaler, ts_scaled = scale_raw(ts)
    
    dim = args.dim
    h = args.step
    dim=168
    h=24

    args.model_name = '{}_h{}_model'.format(data,h)
    dataset = create_dataset(ts, look_back=dim + h - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]

    train_data = dataset[:segmentation,:]
    test_data = dataset[segmentation:,:]

    # train_input = X[:segmentation, :]
    # train_target = Y[:segmentation]
    # test_input = X[segmentation:, :]
    # test_target = Y[segmentation:]
    # train_target = train_target.reshape(-1, h)
    # test_target = test_target.reshape(-1, h)
    
    train_set,_ = prep_data(train_data,train=True,h=h,dim=dim,sample_dense=args.sample_dense)
    test_set, predict_input = prep_data(test_data,train=False,h=h,dim=dim,sample_dense=args.sample_dense) # Remaining modification

    print('')

    # train_rmse_batch = np.empty(10)
    # test_rmse_batch = np.empty(10)
    # train_pred_Batchs = np.empty((train_input.shape[0],h,10))
    # test_pred_Batchs = np.empty((test_input.shape[0],h,10))

    # train_rmse_loss_batch = np.empty((100,10)) 
    # test_rmse_loss_batch = np.empty((100,10))

    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join('experiments', 'model.params.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = Params(json_path)

    params.merge(args)

    params.train_window = dim+h
    params.test_window = params.train_window
    params.predict_start = dim
    # params.test_predict_start=params.predict_start
    params.predict_steps=h
    params.num_class = 1
    params.cov_dim = 0
    params.predict_batch=int(test_set.samples // 2)

    # test
    params.num_epochs = 1

    # params.relative_metrics = args.relative_metrics
    # params.sampling =  args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')


    # create missing directories
    try:
        os.makedirs(params.plot_dir)
    except FileExistsError:
        pass

    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)    

    set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    # logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))

    # model.xfit(train_loader,test_loader,restore_file=None)

    print(predict_input.shape)
    pred = model.point_predict(predict_input)
    target = predict_input[:,params.predict_start:,0]
    vrmse = np.sqrt(mean_squared_error(target,pred))
    print(data + '\t H: ' + str(h) + '\t RMSE: ' + str(vrmse) + '\t')

