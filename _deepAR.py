import os
import numpy as np

# added for preprocessing
import argparse
from data_process.util import Params, set_logger, os_rmdirs
from data_process.metric import rmse, mape, smape
from data_process.dataset import get_dataset, de_scale
from data_process.dataset import deepAR_dataset, deepAR_weight, deepAR_WeightedSampler
from data_process.parser import get_parser


# import torch
import logging

# added for training
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from models.deepAR import DeepAR


def inverse_diff(params, pred, raw_test_data):
    raw_test_target = raw_test_data[:, (0-params.H):].reshape(-1, params.H)
    raw_test_base = raw_test_data[:, (0-params.H-1):-1].reshape(-1, params.H)
    raw_test_pred = pred + raw_test_base

    return raw_test_target, raw_test_pred


def pack_dataset(args):
    json_path = os.path.join('models', 'deepAR.params.json')
    params = Params(json_path)
    params.merge(args)
    dataset_params_path = 'data/paper.esm/lag_settings.json'
    params.update(dataset_params_path)

    params.steps = params.datasets[params.dataset]['lag_order']
    params.normal = params.datasets[params.dataset]['normal']
    params.cov_dim = params.datasets[params.dataset]['cov_dim']

    params.scaler()
    # params.sampling = True

    params.model_name = 'deepAR_{}_h{}'.format(params.dataset, params.H)
    params.model_dir = os.path.join(params.experiment, params.model_name)

    raw_dataset, dataset, train_idx, valid_idx, test_idx = get_dataset(params)
    raw_test_data = raw_dataset[test_idx]
    dataset = params.scaler.fit_transform(dataset)
    train_data, valid_data, test_data = dataset[train_idx], dataset[valid_idx], dataset[test_idx]

    params.train_window = params.steps+params.H
    params.test_window = params.train_window
    params.predict_start = params.steps
    params.predict_steps = params.H

    train_set, train_input, _ = deepAR_dataset(
        train_data, train=True, h=params.H, steps=params.steps, sample_dense=args.sample_dense)
    valid_set, _, _ = deepAR_dataset(
        valid_data, train=True, h=params.H, steps=params.steps, sample_dense=args.sample_dense)
    test_set, test_input, test_target = deepAR_dataset(
        test_data, train=False, h=params.H, steps=params.steps, sample_dense=args.sample_dense)
    test_target = test_target[:, params.predict_start:]

    train_input, train_v = deepAR_weight(train_input, params.steps)
    train_sample = deepAR_WeightedSampler(train_v)

    params.batch_size = 128
    train_loader = DataLoader(
        train_set, batch_size=params.batch_size, sampler=train_sample, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=valid_set.samples,
                              sampler=RandomSampler(valid_set), num_workers=4)

    params.restore = True

    if params.test:
        os_rmdirs(params.model_dir)

    return params, train_loader, valid_loader, test_input, test_target, raw_test_data


if __name__ == "__main__":
    args = get_parser()
    params, train_loader, valid_loader, test_input, test_target, raw_test_data = pack_dataset(
        args)

    times = params.times
    measures = np.zeros((times, 3))
    for i in range(times):
        if os.path.exists(os.path.join(params.model_dir, 'measures.npy')):
            measures = np.load(os.path.join(params.model_dir, 'measures.npy'))
            if measures[i, 0].item() != 0.0:
                continue

        params.cv = i
        logger = logging.getLogger('deepAR.cv{}'.format(i))
        set_logger(os.path.join(params.model_dir,
                                'train.cv{}.log'.format(i)), logger)
        logger.info('Loading the datasets for {}th-batch-training'.format(i))

        model = DeepAR(params, logger)

        logger.info('Loading complete.')
        logger.info(f'Model: \n{str(model)}')

        # if not os.path.exists(os.path.join(params.model_dir, 'best.cv{}.pth.tar'.format(i+1))):
        #     model.xfit(train_loader, valid_loader, restore_file=os.path.join(params.model_dir, 'best.cv{}.pth.tar'.format(i)))

        pred = model.point_predict(test_input)

        if params.diff:
            _test_target, _pred = inverse_diff(params, pred, raw_test_data)
        else:
            # _test_target, _pred = test_target, pred
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        np.save(os.path.join(params.model_dir, 'measures'), measures)
