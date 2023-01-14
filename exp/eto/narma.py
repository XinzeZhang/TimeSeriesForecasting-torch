import os, sys
# print(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from re import S
from numpy.lib.function_base import select
from ray import tune

# from task.ModelSetting import esn_base, cnn_base,esm_base, stat_base
from task.TaskLoader import TaskDataset, Opt
import numpy as np
from task.parser import get_parser
from task.TaskWrapperV1 import Task
# from models.stochastic.cnn import ESM_CNN
import pandas as pd

from models.statistical._setting import autoArima, es , naiveA , naiveL
from models.stochastic.cnn._setting import esm_base, eto_base
from models.stochastic.esn._setting import esn_base, gesn_base, desn_base

class Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    
    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 12
        self.info.period = 10
        self.info.batch_size = 512
        
    def sub_config(self):
        ts_tuple_list = []
        self.info.series_name = ['NARMA10']
        
        for name in self.info.series_name:
            raw_ts = np.load(
                'data/src/narma/narma.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)
        

class esm(esm_base):
    def task_modify(self):
        self.innerTuning = False
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh', 'relu'])
        self.tuning.p_size = tune.qrandint(2, 4, 1)   

class esn(esn_base):        
    def task_modify(self):
        self.tuning.hidden_size = tune.qrandint(50, 500, 25)
        
class gesn(gesn_base):
    def task_modify(self):
        self.tuning.hidden_size = tune.qrandint(5, 50, 5)
        self.tuning.branch_size = tune.qrandint(4, 30, 2)

class desn(desn_base):
    def task_modify(self):
        self.tuning.hidden_size = tune.qrandint(5, 100, 5)
        self.tuning.num_layers = tune.qrandint(2, 16, 2)

class arima(autoArima):        
    def task_modify(self):
        self.hyper.max_length = 12 
        
class holt(es):
    def task_modify(self):
        self.hyper.max_length = 12 


class eto(eto_base):
    def task_modify(self):    
        self.hyper.max_cells = 30
        self.hyper.patience_bos= 10 
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(2, 4, 1)
