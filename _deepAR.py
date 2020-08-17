from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# added for preprocessing
import argparse
from data_process.util import create_dataset,unpadding,scale_raw, Params, set_logger
from data_process.util import deepAR_dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# import torch
import logging

# added for training
import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler

import models.deepAR as net

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


logger = logging.getLogger('DeepAR')

parser = argparse.ArgumentParser(
    description='PyTorch DeeepAR Time Series Forecasting')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-H', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-steps', type=int, default=24)
parser.add_argument('-num_epochs', type=int, default=20, metavar='N',
                    help='epochs for training (default: 20)')

parser.add_argument('-sample-dense', action='store_true',default=True, help='Whether to continually sample the time series during preprocessing')

# parser.add_argument('--model-name', default='brent_model', help='Directory to save model state')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')

if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join('models', 'deepAR.params.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = Params(json_path)

    params.merge(args)
    
    
    #test
    params.dataset = 'MT_001'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    segmentation = int(len(ts)*5/6)
    # ts = ts.reshape(-1,1)
    # scaler, ts_scaled = scale_raw(ts)
    
    params.steps=168
    params.H=24
    # test
    params.num_epochs = 30

    params.model_name = '{}_h{}_model'.format(params.dataset,params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    # X, Y = dataset[:, :(0 - params.H)], dataset[:, (0-params.H):]

    train_data = dataset[:segmentation,:]
    test_data = dataset[segmentation:,:]
    
    train_set,_ = deepAR_dataset(train_data,train=True,h=params.H,steps=params.steps,sample_dense=args.sample_dense)
    test_set, predict_input = deepAR_dataset(test_data,train=False,h=params.H,steps=params.steps,sample_dense=args.sample_dense) # Remaining modification


    params.train_window = params.steps+params.H
    params.test_window = params.train_window
    params.predict_start = params.steps
    # params.test_predict_start=params.predict_start
    params.predict_steps=params.H
    # params.num_class = 1
    params.cov_dim = 0
    params.predict_batch=int(test_set.samples // 2)

    model_dir = os.path.join('experiments', params.model_name)
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    # create missing directories
    try:
        os.makedirs(params.plot_dir)
    except FileExistsError:
        pass
    set_logger(os.path.join(model_dir, 'train.log'),logger)

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

    logger.info('Loading the datasets for batch-training')

    params.batch_size = len(train_set) // 20
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')

    model.xfit(train_loader,test_loader,restore_file=os.path.join(params.model_dir,'best.pth.tar'))

    print(predict_input.shape)
    pred = model.point_predict(predict_input)
    target = predict_input[:,params.predict_start:,0]
    vrmse = np.sqrt(mean_squared_error(target,pred))
    print(params.dataset + '\t H: ' + str(params.H) + '\t RMSE: ' + str(vrmse) + '\t')

