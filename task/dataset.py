# from sklearn.model_selection import TimeSeriesSplit
from asyncio import FastChildWatcher
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from pandas import Series
from torch.utils.data import Dataset, Sampler
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler,SequentialSampler

def difference(dataset, interval=1):
    diff = list()
    diff.append(0)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff).values


def inverse_diff(opts, pred, raw_test_data):
    _pred = de_scale(opts, pred)
    raw_test_target = raw_test_data[:, (0-opts.H):].reshape(-1, opts.H)
    raw_test_base = raw_test_data[:, (0-opts.H-1):-1].reshape(-1, opts.H)
    raw_test_pred = _pred + raw_test_base

    return raw_test_target, raw_test_pred


def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] - 1)

    for i in range(s.shape[0]):
        s[i] = np.diagonal(np.flip(a, 1), offset=-i + h-1,
                           axis1=0, axis2=1).copy().mean()

    return s

# convert an array of values into a dataset matrix





def deepAR_dataset(data, train=True, h=None, steps=None, sample_dense=True):
    assert h != None and steps != None
    raw_data = unpadding(data).reshape(-1, 1)
    time_len = raw_data.shape[0]
    input_size = steps
    window_size = h + steps
    stride_size = h
    if not sample_dense:
        windows_per_series = np.full((1), (time_len-input_size) // stride_size)
    else:
        windows_per_series = np.full((1), 1 + time_len-window_size)
    total_windows = np.sum(windows_per_series)

    x_input = np.zeros((total_windows, window_size, 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    # v_input= np.zeros((total_windows, 2),dtype = 'float32')

    count = 0
    for i in range(windows_per_series[0]):
        # get the sample with minimal time period, in this case. which is 24 points (24h, 1 day)
        stride = 1
        if not sample_dense:
            stride = stride_size

        window_start = stride*i
        window_end = window_start+window_size
        '''
        print("x: ", x_input[count, 1:, 0].shape)
        print("window start: ", window_start)
        print("window end: ", window_end)
        print("data: ", data.shape)
        print("d: ", data[window_start:window_end-1, series].shape)
        '''
        # using the observed value in the t-1 step to forecast the t step, thus the first observed value in the input should be t0 step and is 0, as well as the first value in the labels should be t1 step.

        x_input[count, 1:, 0] = raw_data[window_start:window_end-1, 0]
        label[count, :] = raw_data[window_start:window_end, 0]

        count += 1

    packed_dataset = torch_Dataset(x_data=x_input, label_data=label)
    return packed_dataset, x_input, label


def deepAR_weight(x_batch, steps):
    # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    batch_size = x_batch.shape[0]
    v_input = np.zeros((batch_size, 2), dtype='float32')
    for i in range(batch_size):
        nonzero_sum = (x_batch[i, 1:steps, 0] != 0).sum()
        if nonzero_sum.item() == 0:
            v_input[i, 0] = 0
        else:
            v_input[i, 0] = np.true_divide(
                x_batch[i, 1:steps, 0].sum(), nonzero_sum)+1
            x_batch[i, :, 0] = x_batch[i, :, 0] / v_input[i, 0]

    return x_batch, v_input


class deepAR_WeightedSampler(Sampler):
    def __init__(self, v_input, replacement=True):
        v = v_input.copy()
        self.weights = torch.as_tensor(
            np.abs(v[:, 0])/np.sum(np.abs(v[:, 0])), dtype=torch.double)
        # logger.info(f'weights: {self.weights}')
        self.num_samples = self.weights.shape[0]
        # logger.info(f'num samples: {self.num_samples}')
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


# def de_scale(opts, pred):
#     _pred = pred.copy()
#     ones = np.ones((_pred.shape[0], opts.steps))
#     cat = np.concatenate((ones, _pred), axis=1)
#     _pred = opts.scaler.inverse_transform(cat)[:, -opts.H:]
#     return _pred





# def rnn_dataset(data, h, steps, expand_dim=1):
#     y = data[:, (0-h):].reshape(-1, h)

#     x = np.zeros((data.shape[0], steps, expand_dim))
#     for i in range(expand_dim):
#         x[:, :, i] = data[:, i:steps+i]
#     # return the X with shape: samples, timesteps, dims

#     data_set = torch_Dataset(x_data=x, label_data=y)

#     return data_set, x, y
