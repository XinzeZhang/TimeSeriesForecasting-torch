import os
import numpy as np


from data_process.util import Params, set_logger, os_rmdirs
from data_process.metric import rmse, mape, smape
from data_process.dataset import de_scale, scaled_Dataset, inverse_diff, mlp_dataset, get_dataset
from data_process.parser import get_parser

from sklearn.model_selection import TimeSeriesSplit

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from models.IELM import IELM

import logging

# from pandas import Series

def pack_dataset(args):
    json_path = os.path.join('models', 'SCN.params.json')
    params = Params(json_path)
    params.merge(args)

    dataset_params_path = 'data/paper.esm/lag_settings.json'
    params.update(dataset_params_path)
    params.steps = params.datasets[params.dataset]['lag_order']
    params.normal = params.datasets[params.dataset]['normal']
    # params.hidden_size = (params.steps - params.datasets[params.dataset]['kernel_size']-3 + 2) * 50

    # params.diff = True
    # params.test = True
    # params.normal = False  # False
    # params.search = 'random'
    
    params.scaler()

    params.model_name = 'IELM_{}_h{}'.format(
        params.dataset, params.H)
    params.model_dir = os.path.join(params.experiment, params.model_name)

    raw_dataset, dataset, train_idx, valid_idx, test_idx = get_dataset(params)
    raw_test_data = raw_dataset[test_idx]
    dataset = params.scaler.fit_transform(dataset)
    train_data, valid_data, test_data = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
    # if params.diff:
    #     train_data = train_data[1:,:]

    train_set, _, _ = mlp_dataset(
        train_data, h=params.H, steps=params.steps)
    valid_set, _, _ = mlp_dataset(
        valid_data, h=params.H, steps=params.steps)
    test_set, test_input, test_target = mlp_dataset(
        test_data, h=params.H, steps=params.steps)

    train_loader = DataLoader(train_set, batch_size=train_set.samples)
    valid_loader = DataLoader(valid_set, batch_size=valid_set.samples)

    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')
    
    # params.device = torch.device('cpu')

    params.restore = True


    if params.test:
        os_rmdirs(params.model_dir)
        
    return params, train_loader, valid_loader, test_input, test_target, raw_test_data


if __name__ == "__main__":
    args = get_parser()
    params, train_loader, valid_loader, test_input, test_target, raw_test_data = pack_dataset(args)    


    times = params.times
    measures = np.zeros((times, 3))
    for i in range(times):
        if os.path.exists(os.path.join(params.model_dir, 'measures.npy')):
            measures = np.load(os.path.join(params.model_dir, 'measures.npy'))
            if measures[i, 0].item() != 0.0:
                continue

        params.cv = i
        logger = logging.getLogger('IELM.cv{}'.format(i))
        set_logger(os.path.join(params.model_dir,
                                'train.cv{}.log'.format(i)), logger)
        logger.info(
            'Loading the datasets and model for {}th-batch-training'.format(i))
        # use GPU if available
        model = IELM(params, logger)
        logger.info('Loading complete.')
        logger.info(f'Model: \n{str(model)}')
        if not os.path.exists(os.path.join(params.model_dir, 'train.cv{}.pth.tar'.format(i+1))):
            model.xfit(train_loader, valid_loader)
            pass

        pred = model.predict(test_input)

        if params.diff:
            _test_target, _pred = inverse_diff(params,pred, raw_test_data)
        else:
            _test_target, _pred = de_scale(params, test_target), de_scale(params, pred)

        trmse = rmse(_test_target, _pred)
        tmape = mape(_test_target, _pred)
        tsmape = smape(_test_target, _pred)
        logger.info('{}\t H: {}\t Trail: {} \nTesting RMSE: {:.4g} \t MAPE: {:.4g} \t SMAPE: {:.4g}'.format(
            params.dataset, params.H, i, trmse, tmape, tsmape))
        measures[i, 0] = trmse
        measures[i, 1] = tmape
        measures[i, 2] = tsmape
        np.save(os.path.join(params.model_dir, 'measures'), measures)