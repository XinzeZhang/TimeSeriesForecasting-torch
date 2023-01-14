"""
ARIMA with auto model selection
"""
import os
import sys

# from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
from task.util import unpadding
import pmdarima as pm
# from pmdarima.pipeline import Pipeline
# from pmdarima.preprocessing import BoxCoxEndogTransformer
# from statsmodels.tsa.arima.model import ARIMA as tsaARIMA
from  task.TaskLoader import mlp_dataset, torch_dataloader
from task.TaskLoader import Opt

from tqdm import trange,tqdm
import torch


class arima():
    def __init__(self, opts=None, logger=None):
        self.opts = opts
        self.logger = logger
        
        self.max_length =  opts.lag_order if 'max_length' not in opts.dict else opts.max_length
        self.device = torch.device('cpu')

    def check_history(self,):
        if self.history.shape[0] >  self.max_length:
            self.history = self.history[-self.max_length:]
        
        if self.history.min() < 0:
            self.constant = self.history.min() * -1.05 
        else:
            self.constant = 0
    
        self.history = self.history + self.constant
        if self.history.min() < 0:
            raise ValueError('Current history min() is: {}'.format(self.history.min()))
        return self.history, self.constant

    def data_loader(self, data):
        set_loader = torch_dataloader(mlp_dataset(data, self.opts.H, self.opts.lag_order), batch_size= None,cuda= False)
        return set_loader
        
    def get_history(self, train_data, val_data):
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        batch_x, batch_y = None, None
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.numpy()
            batch_y = batch_y.numpy()

        data_train = np.concatenate((batch_x,batch_y), axis=1)

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.numpy()
            batch_y = batch_y.numpy()

        data_valid = np.concatenate((batch_x,batch_y), axis=1)
        data_fit = np.concatenate((data_train, data_valid), axis=0)
        
        self.data_fit = unpadding(data_fit)
        
        self.history = self.data_fit[:( self.data_fit.shape[0] - (self.opts.H - 1) )]        
        
    def xfit(self, train_data, val_data):
        self.get_history(train_data, val_data)
        self.check_history()
        
        return Opt()
    
    def predict(self, input):
        return input[:, -self.opts.H:]
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        # pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            # batch_pred = self.predict(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            # pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = self.predict(x)
        
        return x, y, pred    
    
    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred

# class pdqARIMA(arima):
#     '''
#     from tsaARIMA
#     '''
#     def __init__(self, opts=None, logger=None):
#         super().__init__(opts, logger)

#         self.p, self.d, self.q = opts.pdq
#         for (arg, value) in opts.dict.items():
#             self.logger.info("Argument %s: %r", arg, value)
    
#     def gen_model(self,):
#         self.model = tsaARIMA(self.history, order = (self.p, self.d, self.q)).fit()
    
#     def predict(self, input):
#         H = self.opts.H
        
#         test_samples = input.shape[0]
    
#         yPred = np.empty((test_samples, H))
#         for i in trange(test_samples):
#             self.gen_model()
#             i_pred = self.model.forecast(H)
#             yPred[i,:]= i_pred - self.constant
#             self.history = np.concatenate((self.history[1:], input[i,-1].reshape(-1,)), axis=0)
#             self.history, self.constant = self.check_history()
#         return yPred

class pdqARIMA(arima):
    '''
    from pmARIMA
    '''
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)

        self.p, self.d, self.q = opts.pdq if 'pdq' in opts.dict else (2,1,2)
        self.P, self.D, self.Q, self.m = opts.PDQM if 'PDQM' in opts.dict else (1,1,1,1)
        
        if self.m == 1:
            self.seasonal = False
        else:
            self.seasonal = True
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
    
    def xfit(self, train_data, val_data):
        self.get_history(train_data, val_data)
        self.check_history()
        self.valid_size = int(self.history.shape[0] * 0.8)
        return Opt()
    
    def gen_model(self,):
        if self.seasonal:
            self.model = pm.ARIMA(order=( self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, self.m), out_of_sample_size=self.opts.H, suppress_warnings=True)
        else:
            self.model = pm.ARIMA(order=( self.p, self.d, self.q),  out_of_sample_size=self.opts.H, suppress_warnings=True)
        self.model.fit(self.history)
        self.logger.info(self.model.summary())

    def predict(self, input):
        H = self.opts.H
        
        test_samples = input.shape[0]        
        yPred = np.empty((test_samples, H))
        
        self.delta_const = 0
        input_min = input.min()
        if input_min + self.constant < 0:
            self.delta_const = (0 - (input_min + self.constant)) * 1.05
        self.history += self.delta_const
        self.gen_model()

        for i in trange(test_samples):
            
            last_input = input[i,-1].reshape(-1,) + self.constant + self.delta_const
            self.model.update(last_input)
            i_pred = self.model.predict(H)
            
            yPred[i,:]= i_pred - self.constant - self.delta_const
        return yPred
    
class autoARIMA(arima):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)
    
        self.max_p, self.d, self.max_q = opts.max_pdq if 'max_pdq' in opts.dict else (5,1, 5)
        self.max_P, self.D, self.max_Q = opts.max_PDQ if 'max_PDQ' in opts.dict else (1,1, 2)
        self.m = opts.m if 'm' in opts.dict else 1
        
        self.seasonal = True if self.m > 1 else False
        
        self.max_iter = 100 if 'max_iter' not in self.opts.dict else opts.max_iter
        self.jobs = 1 if 'jobs' not in self.opts.dict else opts.jobs
        self.stepwise = False if self.jobs > 1 else True
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            

    def xfit(self, train_data, val_data):
        self.get_history(train_data, val_data)
        self.check_history()
        self.valid_size = int(self.history.shape[0] * 0.2)
        return Opt()
    
    def gen_model(self,):
        # self.model = Pipeline([
        #     ("boxcox", BoxCoxEndogTransformer()),
        #     ("model", pm.AutoARIMA(d=self.d, D = self.D, max_p=self.max_p, max_q = self.max_q, maxiter = self.max_iter, n_jobs=self.jobs, stepwise=self.stepwise,  error_action="ignore", out_of_sample_size=self.valid_size, max_P=self.max_P, max_Q=self.max_Q, seasonal=self.seasonal, m=self.m ))
        #     ])
        self.model = pm.AutoARIMA(d=self.d, D = self.D, max_p=self.max_p, max_q = self.max_q, maxiter = self.max_iter, n_jobs=self.jobs, stepwise=self.stepwise,  error_action="ignore", out_of_sample_size=self.valid_size, max_P=self.max_P, max_Q=self.max_Q, seasonal=self.seasonal, m=self.m )
   
        self.model.fit(self.history)
        self.logger.info(self.model.summary())
        # self.model = pm.auto_arima(self.history, max_p=self.max_p, max_q = self.max_q, maxiter = self.max_iter, n_jobs=self.jobs, stepwise=self.stepwise,  error_action="ignore", out_of_sample_size=self.opts.H, max_P=self.max_P, max_Q=self.max_Q, seasonal=self.seasonal, m=self.m)
        
    def predict(self, input):
        H = self.opts.H
        
        test_samples = input.shape[0]        
        # print(history[-3:])
        yPred = np.empty((test_samples, H))
        
        self.delta_const = 0
        input_min = input.min()
        if input_min + self.constant < 0:
            self.delta_const = (0 - (input_min + self.constant)) * 1.05
        self.history += self.delta_const
        self.gen_model()

        for i in trange(test_samples):
            
            last_input = input[i,-1].reshape(-1,) + self.constant + self.delta_const
            self.model.update(last_input)
            i_pred = self.model.predict(H)
            
            yPred[i,:]= i_pred - self.constant - self.delta_const
        return yPred
    