import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
from task.TaskLoader import Opt
from tqdm import tqdm
import torch

from  task.TaskLoader import mlp_dataset, torch_dataloader

class Naive():
    def __init__(self, opts=None, logger=None, method = None):
        self.opts = opts
        self.method = opts.method if method is None else method
        self.logger = logger
        
    def xfit(self, train_loader, val_loader):
        return  Opt()
    
    def by_last(self, input):
        H = self.opts.H
        test_samples = input.shape[0]
        
        yPred = np.empty((test_samples, H))
        
        for i in range(test_samples):
            yPred[i,:] = np.ones((H,)) * input[i, -1]
        
        return yPred
    
    def by_avg(self, input):
        H = self.opts.H
        test_samples = input.shape[0]
        
        yPred = np.empty((test_samples, H))
        for i in range(test_samples):
            yPred[i,:] = np.ones((H,)) * np.average(input[i, :])
        
        return yPred
    
    def by_period(self, input):
        H = self.opts.H
        test_samples = input.shape[0]
        
        yPred = np.empty((test_samples, H))
        for i in range(test_samples):
            yPred[i,:] = input[i, -H:]
        
        return yPred
    
    def data_loader(self, data):
        set_loader = torch_dataloader(mlp_dataset(data, self.opts.H, self.opts.lag_order), batch_size= None,cuda= False)
        return set_loader
    
    def predict(self, input):
        if self.method == 'last':
            yPred = self.by_last(input)
        elif self.method == 'avg':
            yPred = self.by_avg(input)
        elif self.method == 'period':
            yPred = self.by_period(input)
        else:
            raise ValueError('Non supported method: {}'.format(self.method))
        
        return yPred
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        # pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.cpu()
            batch_y = batch_y.cpu()
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
    
# class Naive_period(Naive):
#     def __init__(self, opts=None, logger=None):
#         super().__init__(opts, logger)
        
#     def xfit(self, train_loader, val_loader):
#         return  0.01
    
#     def predict(self, input):
#         H = self.opts.H
#         test_samples = input.shape[0]
        
#         yPred = np.empty((test_samples, H))
#         for i in range(test_samples):
#             yPred[i,:] = input[i, -H:]
        
#         return yPred
    
# class Naive_random(Naive):
#     def __init__(self, opts=None, logger=None):
#         super().__init__(opts, logger)
        
#     def xfit(self, train_loader, val_loader):
#         return  0.01
    
#     def predict(self, input):
#         H = self.opts.H
#         test_samples = input.shape[0]
        
#         yPred = np.empty((test_samples, H))
        
#         for i in range(test_samples):
#             yPred[i,:] = input[i, -H:]
            
#         random_v = np.random.normal(size=(test_samples, H))
        
#         yPred = yPred + random_v
        
#         return yPred    