import os
import numpy as np

# added for preprocessing
from data_process.util import Params, set_logger, os_rmdirs
from data_process.metric import rmse, mape, smape
from data_process.dataset import de_scale, inverse_diff, cnn_dataset, get_dataset
from data_process.parser import get_parser

import torch
from torch.utils.data import DataLoader
from models.ESM_CNN import ESM_CNN

import logging

def pack_dataset(args):
    json_path = os.path.join('models', 'ESC_CNN.params.json')
    params = Params(json_path)
    params.merge(args)

    dataset_params_path = 'data/{}/lag_settings.json'.format(params.datafolder)
    params.update(dataset_params_path)
    params.steps = params.datasets[params.dataset]['lag_order']
    params.normal = params.datasets[params.dataset]['normal']
    params.cov_dim =params.datasets[params.dataset]['cov_dim']
    params.kernel_size = params.datasets[params.dataset]['kernel_size']
    params.kernel_list = [int(params.steps / 3), int(params.steps / 4), int(params.steps / 5), int(params.steps / 6)]

    # params.test = True
    # params.diff = True
    # params.normal = False  # False
    params.search = 'random'
    params.scaler()

    params.model_name = 'esmCNN_{}_h{}'.format(
        params.dataset, params.H)
    params.model_dir = os.path.join(params.experiment, params.model_name)

    raw_dataset, dataset, train_idx, valid_idx, test_idx = get_dataset(params)
    raw_test_data = raw_dataset[test_idx]
    dataset = params.scaler.fit_transform(dataset)
    train_data, valid_data, test_data = dataset[train_idx], dataset[valid_idx], dataset[test_idx]

    train_set, _, _ = cnn_dataset(
        train_data, h=params.H, steps=params.steps)
    valid_set, _, _ = cnn_dataset(
        valid_data, h=params.H, steps=params.steps)
    test_set, test_input, test_target = cnn_dataset(
        test_data, h=params.H, steps=params.steps)

    params.In_channels = 1 + params.cov_dim

    train_loader = DataLoader(train_set, batch_size=train_set.samples)
    valid_loader = DataLoader(valid_set, batch_size=valid_set.samples)

    
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
        logger = logging.getLogger('esmCNN.cv{}'.format(i))
        set_logger(os.path.join(params.model_dir,
                                'train.cv{}.log'.format(i)), logger)
        logger.info(
            'Loading the datasets and model for {}th-batch-training'.format(i))
        # use GPU if available
        model = ESM_CNN(params, logger)
        logger.info('Loading complete.')
        logger.info('Model: \n{}'.format(str(model)))
        if not os.path.exists(os.path.join(params.model_dir, 'train.cv{}.pth.tar'.format(i+1))):
            model.xfit(train_loader, valid_loader, restore_file=os.path.join(
                params.model_dir, 'train.cv{}.pth.tar'.format(i)))

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