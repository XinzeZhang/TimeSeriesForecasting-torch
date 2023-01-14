import os

from numpy.lib.function_base import select

# from torch.utils import data

# from task.util import os_rmdirs, set_logger

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import numpy as np

# from models.Config import model_opts_dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from collections.abc import Mapping
import copy

# from task.metric import rmse

class Opt(object):
    def __init__(self, init = None):
        super().__init__()
        
        if init is not None:
            self.merge(init)
        
    def merge(self, opts, ele_s = None):
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        
        if ele_s is None:
            for key in new:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key]) 
        else:
            for key in ele_s:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key])
                
    def update(self, opts, ignore_unk = False):
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        for key in new:
            if not key in self.dict and ignore_unk is False:
                raise ValueError(
                "Unknown config key '{}'".format(key))
            self.dict[key] = copy.copy(new[key])
            
    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__

class Params(Opt):
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, args = None,):
        if args is not None:
            self.merge(args)        
        # if json_path is not None:
        #     with open(json_path) as f:
        #         params = json.load(f)
        #         assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
        #         self.__dict__.update(params)
    # def update(self, dict):
    #     for key in dict:
    #         if not key in self.dict:
    #             raise ValueError(
    #             "Unknown config key '{}'".format(key))
    #         self.dict[key] = dict[key]

class TaskDataset(Opt):
    def __init__(self, args):
        super().__init__()
        self.info = Opt()
        self.info.normal = args.normal
        self.batch_size = None
        self.info_config()
        self.info.H = args.H
        self.args = args
        self.sub_config()
        
        self.info.num_series = len(self.seriesPack)
        if 'series_name' in self.info.dict:
            assert len(self.info.series_name) == self.info.num_series
            
    
    def info_config(self, ):
        pass
    
    def sub_config(self,):
        pass
    
    def pack_subs(self, ts_tuple_list):
        self.seriesPack = []
        for i, ts_tuple in enumerate(ts_tuple_list):
            ts_name = ts_tuple[0]
            ts = ts_tuple[1]
            sub = self.pack_dataset(raw_ts=ts, index=i, name=ts_name)
            self.seriesPack.append(sub)
    
    
    def pack_dataset(self, raw_ts, index = 0, name = 'default', H = None,  test = False ):
        
        sub = Opt()
       
        _dataset, train_idx, valid_idx, test_idx = self.data_split(raw_ts)
        
        sub.scaler = Scaler(self.info.normal)
        sub.scaler.fit(raw_ts)
        
        #for testing
        # sub.orig = _dataset
        
        # dataset = sub.scaler.transform(_dataset)
        if test:
            dataset = _dataset # only for testing
        else:
            dataset = sub.scaler.transform(_dataset)
        
        train_data, valid_data, test_data = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
                
        # train_loader, valid_loader, test_loader = self.data_loader(dataset[train_idx], dataset[valid_idx], dataset[test_idx])
                
        sub.train_data = train_data
        sub.valid_data = valid_data
        sub.test_data = test_data
        sub.lag_order = self.info.lag_order
        sub.index = index
        sub.name = name
        if H is None:
            sub.H = self.info.H
        else:
            sub.H = H
        sub.merge(self.info)
        
        # sub.test_input = test_input
        # sub.test_target = test_target
        # to do check the torch.cat([batch_x, for b_x in test_loader]) is test_input
        # to do replace, model.predict with model.loader_pre    
        
        # generate the results of naive forecasting method.
        last_naive, avg_naiveE = self.get_naive_pred(sub)
        sub.lastNaive = last_naive
        sub.avgNaiveError = avg_naiveE
    
        return sub
    
    def get_naive_pred(self, sub):
        
        
        _tx = sub.test_data[:,:sub.lag_order]
        last_naive = _tx[:, -1]
        
        avg_naiveE = np.zeros_like(last_naive)
        lag_order = _tx.shape[1]
        for t in range(1, lag_order):
            step_error = _tx[:,t] - _tx[:, t-1]
            step_error = np.abs(step_error)
            avg_naiveE += step_error
        
        avg_naiveE = avg_naiveE / (lag_order - 1)
        
        return last_naive, avg_naiveE
        
    def data_split(self,raw_ts):
        # print(self.info.steps)
        # print(self.info.H)
        dataset = create_dataset(raw_ts, self.info.lag_order + self.info.H)
        
        tscv = TimeSeriesSplit(n_splits=self.args.k-1)
        *lst, last = tscv.split(dataset)
        train_idx, test_idx = last
        _train = dataset[train_idx]
        train_tscv = TimeSeriesSplit(n_splits=self.args.k-1)
        *lst, last = train_tscv.split(_train)
        train_idx, valid_idx = last
        
        return dataset, train_idx, valid_idx, test_idx

class Scaler:
    def __init__(self, normal):
        if normal:
            self.scaler = StandardScaler()
        else:
            self.scaler= MinMaxScaler(feature_range=(-1, 1))
            
    def fit(self, dataset):
        # check 1d
        assert len(dataset.shape) == 1
        data = dataset.reshape(-1,1)
        self.scaler.fit(data)
        
    def transform(self, dataset):
        # check 2d
        assert len(dataset.shape)  == 2
        
        temp = np.empty_like(dataset)
        
        for c in range(dataset.shape[1]):
            column = dataset[:,c].reshape(-1,1)
            column = self.scaler.transform(column).reshape(-1,)
            temp[:,c]= column
        
        return temp
            
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
        
    
    def inverse_transform(self, dataset):
        assert len(dataset.shape)  == 2
        
        temp = np.empty_like(dataset)
        
        for c in range(dataset.shape[1]):
            column = dataset[:,c].reshape(-1,1)
            column = self.scaler.inverse_transform(column).reshape(-1,)
            temp[:,c]= column
        
        return temp

class torch_Dataset(Dataset):
    '''
    Packing the input x_data and label_data to torch.dataset
    '''
    def __init__(self, x_data, label_data):
        self.data = np.float32(x_data.copy())
        self.label = np.float32(label_data.copy())
        self.samples = self.data.shape[0]
        # logger.info(f'samples: {self.samples}')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

def create_dataset(dataset, sample_lens):

    # dataset = np.insert(dataset, [0] * look_back, 0)
    data = []
    for i in range(len(dataset) - sample_lens + 1):
        a = dataset[i:(i + sample_lens )]
        data.append(a)

    dataset = np.array(data)
    # dataY = np.reshape(dataY, (dataY.shape[0], 1))
    # dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset
    
def torch_dataloader(torchData, batch_size = None, cuda = False):
    if batch_size is not None:
        _batch_size  = torchData.samples if torchData.samples < batch_size else batch_size
    else:
        _batch_size = torchData.samples
    
    set_loader = DataLoader(torchData, batch_size=_batch_size, pin_memory=cuda, shuffle=False, sampler=SequentialSampler(torchData))
    return set_loader

def rnn_dataset(data, h, l, d, sso = False):
    assert data.shape[1] == l + h 
    assert d < l
    assert d+h < data.shape[1]
    
    x = []
    y = []
    for row_idx in range(data.shape[0]):
        row = data[row_idx,:]
        # print(row)
        _h = 1 if sso else h
        data_r = create_dataset(row, d+_h)
        data_x = data_r[:,:d]
        data_y = data_r[:, -_h:]
        x.append(np.transpose(data_x))
        y.append(np.transpose(data_y))
        
        # print(x[row_idx])
        # print(y[row_idx])
    
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    try:
        assert x.shape[0] == data.shape[0]
        assert x.shape[1] == d
        if sso:
            assert x.shape[2] == (l + h - d )
        else:
            assert x.shape[2] == (l - d + 1)
    except:
            raise ValueError('x shape: {}'.format(x.shape)+'\nl: {}\th: {}\td: {}'.format(l,h,d))

    data_set = torch_Dataset(x_data=x, label_data=y)

    return data_set

def mlp_dataset(data, h, l):
    x = data[:, :(0 - h)].reshape(data.shape[0], l)
    y = data[:, (0-h):].reshape(-1, h)
    data_set = torch_Dataset(x_data=x, label_data=y)

    return data_set

def dnn_dataset(data, h, l, d):
    '''
    x, shape: (N_samples, dimensions(input_dim), steps)\n
    y, shape: (N_samples, dimensions)
    '''
    assert data.shape[1] == l + h 
    assert d < l
    assert d+h < data.shape[1]
    
    x = []
    y = []
    for row_idx in range(data.shape[0]):
        row = data[row_idx,:]
        # print(row)

        data_r = create_dataset(row, d+h)
        data_x = data_r[:,:d]
        data_y = data_r[-1, -h:]
        x.append(np.transpose(data_x))
        y.append(np.transpose(data_y))
        
        # print(x[row_idx])
        # print(y[row_idx])
    
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    try:
        assert x.shape[0] == data.shape[0]
        assert x.shape[1] == d
        assert x.shape[2] == (l - d + 1)

    except:
            raise ValueError('x shape: {}'.format(x.shape)+'\nl: {}\th: {}\td: {}'.format(l,h,d))

    data_set = torch_Dataset(x_data=x, label_data=y)

    return data_set
