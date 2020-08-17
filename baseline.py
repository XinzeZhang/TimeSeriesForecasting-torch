from data_process.util import create_dataset, scale_raw
import argparse
from model.MSVR import MSVR

from numpy import concatenate, atleast_2d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(
    description='Time Series Forecasting Baseline model')
parser.add_argument('-step', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-dim', type=int, default=24)

if __name__ == "__main__":
    args = parser.parse_args()

    data = args.dataset
    data = 'MT_001'
    ts = np.load('./data/paper/{}.npy'.format(data))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    segmentation = int(len(ts)*5/6)
    # ts = ts.reshape(-1,1)
    # scaler, ts_scaled = scale_raw(ts)

    dim = args.dim
    h = args.step
    dim = 168
    h = 24

    args.model_name = '{}_h{}_model'.format(data, h)
    dataset = create_dataset(ts, look_back=dim + h - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]
    train_input = X[:segmentation, :]
    train_target = Y[:segmentation]
    test_input = X[segmentation:, :]
    test_target = Y[segmentation:]

    train_target = train_target.reshape(-1, h)
    test_target = test_target.reshape(-1, h)

    msvr = MSVR(epsilon=0.01, C=1000, gamma=0.01, tol=0.0001)
    rmse, vrmse, pred, vpred = msvr.fit_validate(
        train_input, train_target, test_input, test_target)
    print(data + '\t H: ' + str(h) + '\t RMSE: ' + str(vrmse) + '\t')
