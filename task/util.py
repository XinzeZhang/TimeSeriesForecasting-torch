import os
import sys

# from sklearn.utils import gen_even_slices
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import math
from torch.utils.data import Dataset, Sampler

import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import torch
import shutil
import logging
import json
from functools import reduce
from pathlib import Path
import shutil
import pandas as pd

matplotlib.use('Agg')
#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions


# logger = logging.getLogger('DeepAR.Data')


# def scale_raw(raw):
#     # fit scaler
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     # raw = raw.reshape(raw.shape[0], 1)
#     scaler = scaler.fit(raw)
#     # transform train
#     norm_raw = scaler.transform(raw)
#     norm_raw = norm_raw[:, 0]
#     return scaler, norm_raw

def class_exist(str):
    _exist = False
    try:
        _exist = reduce(getattr, str.split("."), sys.modules[__name__])
    except:
        pass
    
    return _exist

def toTorch(train_input, train_target, test_input, test_target):
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()
    return train_input, train_target, test_input, test_target

def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] - 1)

    for i in range(s.shape[0]):
        s[i] = np.diagonal(np.flip(a, 1), offset=-i + h-1,
                           axis1=0, axis2=1).copy().mean()

    return s
    

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

def scaler_inverse(scaler, data):
    assert len(data.shape) == 2
    _data = scaler.inverse_transform(data)
    return _data

def IQR_check(data, ids = None, r=1.5, l_b = True, u_b = True):
    '''data shape: (samples,)
    '''
    if ids == None:
        tag = list(range(data.shape[0]))
    else:
        tag = ids
    eval_df = pd.DataFrame({'m': data, 'tag':tag})
    
    eval_q1 = eval_df['m'].quantile(0.25, interpolation='nearest')
    eval_q3 = eval_df['m'].quantile(0.75, interpolation='nearest')
    eval_iqr = eval_q3 - eval_q1
    eval_l_bound = eval_q1 - r * eval_iqr
    eval_u_bound = eval_q3 + r * eval_iqr
    if l_b and u_b:
        outlier_ids = eval_df[(eval_df['m'] < eval_l_bound) | (eval_df['m'] > eval_u_bound)].index.tolist()
    elif l_b and u_b is False:
        outlier_ids = eval_df[eval_df['m'] > eval_u_bound].index.tolist()
    elif l_b is False and u_b:
        outlier_ids = eval_df[eval_df['m'] < eval_l_bound].index.tolist()
    else:
        outlier_tag = []
        return outlier_tag 
    
    outlier_tag =  eval_df['tag'].loc[outlier_ids].values.tolist()
    
    return outlier_tag
# def unpadding(y):
#     a = y.copy()
#     h = y.shape[1]
#     s = np.empty(y.shape[0] + y.shape[1] - 1)

#     for i in range(s.shape[0]):
#         s[i] = np.diagonal(np.flip(a, 1), offset=-i + h-1,
#                            axis1=0, axis2=1).copy().mean()

#     return s
def if_choose(data, if_op):
    '''
    data: numpy array (sample, dims, steps)\n
    if_op: list lens== steps
    '''
    _data = []
    for id, tag in enumerate(if_op):
        if int(tag) == 1:
            _data.append(data[:,:,id])
    if_data = np.stack(_data, axis=-1)
    return if_data

def if_check(opts, length):
    input_select = False
    for i in range(length):
        if 'if_strip_{}'.format(i) in opts.dict:
            input_select = True
            break
    
    if input_select:
        for i in range(length):
            if 'if_strip_{}'.format(i) not in opts.dict:
                raise ValueError('Missing hyper config: "strip_{}"!'.format(i))
    
    return input_select  


class torch_Dataset(Dataset):
    '''
    Packing the input x_data and label_data to torch.dataset
    '''

    def __init__(self, x_data, label_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.samples = self.data.shape[0]
        # logger.info(f'samples: {self.samples}')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (self.data[index, :, :], self.label[index])


class multiClass_Dataset(Dataset):
    '''
    only support multiple class
    '''

    def __init__(self, x_data, label_data, v_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.v = v_data.copy()
        self.test_len = self.data.shape[0]
        # logger.info(f'test_len: {self.test_len}')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])


# class _deepAR_dataset(Dataset):
#     def __init__(self, x_data, label_data, v_data):
#         self.data = x_data.copy()
#         self.label = label_data.copy()
#         self.v = v_data.copy()
#         self.samples = self.data.shape[0]

#     def __len__(self):
#         return self.samples

#     def __getitem__(self, index):
#         return (self.data[index,:,:],self.v[index],self.label[index])

# def deepAR_dataset(data, train=True, h=None, steps=None, sample_dense=True):
#     assert h != None and steps != None
#     raw_data = unpadding(data).reshape(-1, 1)
#     time_len = raw_data.shape[0]
#     input_size = steps
#     window_size = h + steps
#     stride_size = h
#     if not sample_dense:
#         windows_per_series = np.full((1), (time_len-input_size) // stride_size)
#     else:
#         windows_per_series = np.full((1), 1 + time_len-window_size)
#     total_windows = np.sum(windows_per_series)

#     x_input = np.zeros((total_windows, window_size, 1), dtype='float32')
#     label = np.zeros((total_windows, window_size), dtype='float32')
#     # v_input= np.zeros((total_windows, 2),dtype = 'float32')

#     count =0
#     for i in range(windows_per_series[0]):
#         # get the sample with minimal time period, in this case. which is 24 points (24h, 1 day)
#         stride = 1
#         if not sample_dense:
#             stride = stride_size

#         window_start = stride*i
#         window_end = window_start+window_size
#         '''
#         print("x: ", x_input[count, 1:, 0].shape)
#         print("window start: ", window_start)
#         print("window end: ", window_end)
#         print("data: ", data.shape)
#         print("d: ", data[window_start:window_end-1, series].shape)
#         '''
#         # using the observed value in the t-1 step to forecast the t step, thus the first observed value in the input should be t0 step and is 0, as well as the first value in the labels should be t1 step.

#         x_input[count, 1:, 0] = raw_data[window_start:window_end-1, 0]
#         label[count, :] = raw_data[window_start:window_end, 0]
        
#         count += 1

#     packed_dataset = torch_Dataset(x_data=x_input, label_data=label)
#     return packed_dataset, x_input, label

# def deepAR_weight(x_batch, steps):
#     # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
#     batch_size = x_batch.shape[0]
#     v_input= np.zeros((batch_size, 2),dtype ='float32')
#     for i in range(batch_size):
#         nonzero_sum = (x_batch[i, 1:steps, 0]!=0).sum()
#         if nonzero_sum.item() == 0:
#             v_input[i,0]=0
#         else:
#             v_input[i,0]=np.true_divide(x_batch[i, 1:steps, 0].sum(),nonzero_sum)+1
#             x_batch[i,:,0] = x_batch[i,:,0] / v_input[i,0]
        
#     return x_batch, v_input

# class deepAR_WeightedSampler(Sampler):
#     def __init__(self, v_input, replacement=True):
#         v = v_input.copy()
#         self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
#         # logger.info(f'weights: {self.weights}')
#         self.num_samples = self.weights.shape[0]
#         # logger.info(f'num samples: {self.num_samples}')
#         self.replacement = replacement

#     def __iter__(self):
#         return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

#     def __len__(self):
#         return self.num_samples


def set_logger(log_path, log_name, level = 20, rewrite = True):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `task_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    
    logger = logging.Logger(log_name)
    if os.path.exists(log_path) and rewrite:
        os.remove(log_path) # os.remove can only delete a file with given file_path; os.rmdir() can delete a directory.
    log_file = Path(log_path)
    log_folder = log_file.parent
    os_makedirs(log_folder)
    log_file.touch(exist_ok=True)


    if level == 50:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(TqdmHandler(fmt))

    return logger

def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(
            checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        print(
            f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')
    print('Checkpoint saved to {}'.format(filepath))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        # logger.info('Best checkpoint copied to best.pth.tar')
        print('Best checkpoint copied to best.pth.tar')


def savebest_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(checkpoint, 'best.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.epochs -= checkpoint['epoch'] + 1
        if model.epochs < 0:
            model.epochs = 0

    return checkpoint


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()

def plot_xfit(fit_info, save_name, location='./figures/'):
    os_makedirs(location)
    tloss,vloss = np.array(fit_info.loss_list), np.array(fit_info.vloss_list)
    num_samples = tloss.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, tloss[:num_samples], label='Training')
    plt.plot(x, vloss[:num_samples], label='Validation')
    plt.legend()
    f.savefig(os.path.join(location, save_name + '.xfit.png'))
    np.save(os.path.join(location, save_name) + '.loss', (tloss, vloss))
    plt.close()

def plot_hError(H_error, metrics, cid = 0, location = './figures/'):
    
    os_makedirs(location)
    x = np.arange(start=1, stop=H_error.shape[0] + 1)
    
    for i, m in enumerate(metrics):
        f = plt.figure()
        plt.plot(x, H_error[:, i], label = m)
        plt.legend()
        
        # x_range = np.arange(1, horizon + 1)
        # plt.xticks(x_range)
        f.savefig(os.path.join(location, '{}.cv{}.step.error.png'.format(m, cid)))
        plt.close()
        # np.save(os.path.join(location, m + '.step.error'), H_error[:,i])
    
    
    
def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                           predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                           alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})

        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def de_scale(params, pred):
    _pred = params.scaler.inverse_transform(pred)
    return _pred

def os_makedirs(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except FileExistsError:
        pass

def os_rmdirs(folder_path):
    try:
        dirPath = Path(folder_path)
        if dirPath.exists() and dirPath.is_dir():
            shutil.rmtree(dirPath)
    except FileExistsError:
        pass