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
from sklearn.model_selection import KFold

# import torch
import logging

# added for training
import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler

import models.deepAR as net
from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(
    description='PyTorch DeeepAR Time Series Forecasting')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-H', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-steps', type=int, default=24)
parser.add_argument('-num_epochs', type=int, default=20, metavar='N',
                    help='epochs for training (default: 20)')
parser.add_argument('-k', type=int, default=5)

parser.add_argument('-sample-dense', action='store_true',default=True, help='Whether to continually sample the time series during preprocessing')
parser.add_argument('-restore', action='store_true', help='Whether to restore the model state from the best.pth.tar')

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
    
    
    params.dataset = 'london_2013_summary'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    
    params.steps=168
    params.H=24
    # test
    params.train_window = params.steps+params.H
    params.test_window = params.train_window
    params.predict_start = params.steps
    # params.test_predict_start=params.predict_start
    params.predict_steps=params.H
    # params.num_class = 1
    params.cov_dim = 0

    params.num_epochs = 30


    params.model_name = '{}_h{}_deepAR'.format(params.dataset,params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    kf = KFold(n_splits=params.k)
    kf.get_n_splits(dataset)
    params.restore = False

    cvs = []
    for i, (train_idx, test_idx) in tqdm(enumerate(kf.split(dataset))):
        params.cv = i
        logger = logging.getLogger('deepAR.cv{}'.format(i))
        model_dir = os.path.join('experiments', params.model_name)
        set_logger(os.path.join(model_dir, 'train.cv{}.log'.format(i)), logger)

        train_data = dataset[train_idx]
        test_data = dataset[test_idx]
    
        train_set,_ = deepAR_dataset(train_data,train=True,h=params.H,steps=params.steps,sample_dense=args.sample_dense)
        test_set, predict_input = deepAR_dataset(test_data,train=False,h=params.H,steps=params.steps,sample_dense=args.sample_dense) # Remaining modification

        logger.info('Loading the datasets for batch-training')
        params.batch_size = len(train_set) // 4
        train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
        test_loader = DataLoader(test_set, batch_size=test_set.samples,
                                sampler=RandomSampler(test_set), num_workers=4)

        # params.predict_batch=int(test_set.samples)

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
            model = net.Net(params, logger).cuda()
        else:
            params.device = torch.device('cpu')
            # torch.manual_seed(230)
            logger.info('Not using cuda...')
            model = net.Net(params,logger)    


        logger.info('Loading complete.')

        logger.info(f'Model: \n{str(model)}')

        model.xfit(train_loader,test_loader,restore_file=os.path.join(params.model_dir,'best.pth.tar'))

        print(predict_input.shape)
        pred = model.point_predict(predict_input)
        target = predict_input[:,params.predict_start:,0]
        cvs.append((pred, target))

    measures = np.zeros(len(cvs))
    for i, (pred, target) in enumerate(cvs):
        vrmse = np.sqrt(mean_squared_error(target, pred))
        measures[i] = vrmse
        logger.info('{}\t H: {}\t CV: {}\t RMSE:{}'.format(
            params.dataset, params.H, i, vrmse))
        logger.info(params.dataset + '\t H: ' + str(params.H) +
              '\t Avg RMSE: ' + str(measures.mean()) + '\t')

