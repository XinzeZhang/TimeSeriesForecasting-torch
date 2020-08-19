import logging
from sklearn.model_selection import KFold
from tqdm import tqdm
from data_process.util import create_dataset, Params, de_scale, scaled_Dataset, set_logger
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from models.CNN import CNN

from numpy import concatenate, atleast_2d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(
    description='Time Series Forecasting')
parser.add_argument('-step', type=int, default=1, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-steps', type=int, default=24)
parser.add_argument('-k', type=int, default=5)

# parser.add_argument('-in_channels', type=int, default=1)
parser.add_argument('-channel_size', type=int, default=300)
parser.add_argument('-restore', action='store_true', help='Whether to restore the model state from the best.pth.tar')

if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join('models', 'convRNN.params.json')
    assert os.path.isfile(
        json_path), f'No json configuration file found at {json_path}'
    params = Params(json_path)

    params.merge(args)

    #test
    params.dataset = 'AR1'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    # segmentation = int(len(ts)*5/6)
    params.steps = 15
    params.H = 3
    # test

    params.model_name = '{}_h{}_CNN'.format(params.dataset, params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    kf = KFold(n_splits=params.k)
    kf.get_n_splits(dataset)

    cvs = []
    for i, (train_idx, test_idx) in tqdm(enumerate(kf.split(dataset))):
        train_data = dataset[train_idx]
        test_data = dataset[test_idx]

        x_train = train_data[:, :(
            0 - params.H)].reshape(train_data.shape[0], 1, params.steps)
        y_train = train_data[:, (0-params.H):].reshape(-1, params.H)
        x_test = test_data[:, :(0 - params.H)
                           ].reshape(test_data.shape[0], 1, params.steps)
        y_test = test_data[:, (0-params.H):].reshape(-1, params.H)


        params.in_channels = x_train.shape[1]
        train_set = scaled_Dataset(x_data=x_train, label_data=y_train)
        test_set = scaled_Dataset(x_data=x_test, label_data=y_test)

        # added for testing
        params.batch_size = train_set.samples
        params.restore = False
        # params.test_batch=int()

        train_loader = DataLoader(
            train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
        val_loader = DataLoader(test_set, batch_size=test_set.samples,
                                sampler=RandomSampler(test_set), num_workers=4)

        model_dir = os.path.join('experiments', params.model_name)
        params.model_dir = model_dir
        params.plot_dir = os.path.join(model_dir, 'figures')
        # create missing directories
        try:
            os.makedirs(params.plot_dir)
        except FileExistsError:
            pass

        logger = logging.getLogger('CNN.cv{}'.format(i))
        set_logger(os.path.join(model_dir, 'train.cv{}.log'.format(i)), logger)

        # use GPU if available
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda')
            # torch.cuda.manual_seed(240)
            logger.info('Using Cuda...')
            model = CNN(params, logger).cuda()
        else:
            params.device = torch.device('cpu')
            # torch.manual_seed(230)
            logger.info('Not using cuda...')
            model = CNN(params, logger)

        logger.info('Loading the datasets for batch-training')

        logger.info('Loading complete.')

        logger.info(f'Model: \n{str(model)}')

        params.num_epochs = 100
        model.xfit(train_loader, val_loader, restore_file=os.path.join(
            params.model_dir, 'best.pth.tar'))

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
