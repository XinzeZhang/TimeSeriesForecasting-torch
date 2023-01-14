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


from models.stochastic.cnn._setting import eto_base


class Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        # self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.lag_order = 180
        # self.info.input_dim = 1
        self.info.period = 60
        self.info.batch_size = 4096
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['laser.D']

        for name in self.info.series_name:            
            raw_ts = np.load(
                'data/synthetic/laser/laser.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)
        

class eto(eto_base):
    def task_modify(self):    
        self.hyper.max_cells = 50
        self.hyper.patience_bos = 40
        self.hyper.esn.tuning.hidden_size =  tune.qrandint(5, 50, 5)
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        self.hyper.esc.tuning.esn_hidden_size = tune.qrandint(5, 50, 5)
        self.hyper.esc.tuning.kernel_size = tune.qrandint(30, 90, 10)
        self.hyper.ces.tuning.esn_hidden_size = tune.qrandint(5, 50, 5)
        self.hyper.ces.tuning.kernel_size = tune.qrandint(30, 90, 10)
        
class pt(eto):
    def ablation_modify(self):
        self.hyper.cTrain = False
        self.hyper.rTune = False
        
class st(eto):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
        self.hyper.esn.tuner.iters = 1
        self.hyper.esc.tuner.iters = 1
        self.hyper.ces.tuner.iters = 1
        
        self.hyper.rTune = False
        
class rt(eto):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
        self.hyper.esn.tuner.iters = 1
        self.hyper.esc.tuner.iters = 1
        self.hyper.ces.tuner.iters = 1
        
        self.hyper.cTrain = False
        
class ps(eto):
    def ablation_modify(self):
        self.hyper.rTune = False
        
class pr(eto):
    def ablation_modify(self):
        self.hyper.cTrain = False
        
class sr(eto):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
        self.hyper.esn.tuner.iters = 1
        self.hyper.esc.tuner.iters = 1
        self.hyper.ces.tuner.iters = 1