import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import concatenate
import torch.nn.init as Init
# from data_process._data_process import mape

from data_process.util import savebest_checkpoint, plot_all_epoch, plot_xfit,load_checkpoint
from data_process.util import  os_makedirs

import os

def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    if model.params.device == torch.device('cpu'):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint, map_location='cuda')

    model.hidden_size = checkpoint['hidden_size']
    model.weight_IH = checkpoint['weight_IH']
    model.bias_IH = checkpoint['bias_IH']
    model.weight_HO = checkpoint['weight_HO']

def save_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(
        checkpoint, 'train.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)

class S_CNN(nn.Module):
    def __init__(self, params=None, logger = None):
        super(S_CNN, self).__init__()
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.In_channels = 1 + params.cov_dim
        self.Timesteps = params.steps
        self.Output_dim = params.H
        # self.Output_dim = output_dim
        self.Channel_size = params.channel_size
        self.Kernel_size = params.kernel_size
        self.p_size = 3

        with torch.no_grad():
            self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,self.Kernel_size,padding=0)
            self.Pool = nn.AvgPool1d(kernel_size=self.p_size,stride=1,padding=0)

            Lambda = params.Lambdas[0]
            weight_candidate = torch.empty(
                self.Channel_size, self.In_channels, self.Kernel_size).uniform_(-Lambda, Lambda).float().to(self.params.device)
            bias_candidates = torch.empty(
                self.Channel_size).uniform_(-Lambda, Lambda).float().to(self.params.device)
            self.Conv.weight.data = weight_candidate
            self.Conv.bias.data = bias_candidates
        self.out_weight = None

        self.loss_fn = nn.MSELoss()

        self.params.plot_dir = os.path.join(params.model_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

    def solve_output(self, feature, target):
        with torch.no_grad():
            C_state = torch.cat(
                (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
            output_w, _ = torch.lstsq(target, C_state)
            output_w = output_w[0:C_state.size(1)].to(self.params.device)
        return output_w

    def forward(self,input):
        # input = input.permute(0, 2, 1)
        feature_map = self.Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        
        feature_map = feature_map.view(-1, self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2))
        
        feature_map = torch.cat(
        (torch.ones(feature_map.size(0), 1).to(self.params.device), feature_map), 1)
        
        pred = feature_map.mm(self.out_weight)

        return pred
    
    def xfit(self, train_loader, val_loader, restore_file = None):
        with torch.no_grad():
            min_vmse = 9999
            train_x, train_y = None, None
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()

            val_x, val_y = None, None
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                val_x, val_y = batch_x.detach().clone(), batch_y.detach().clone()

            feature_map = self.Conv(train_x)
            feature_map = torch.sigmoid(feature_map)
            feature_map = self.Pool(feature_map)
            
            feature_map = feature_map.view(-1, self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2))

            self.out_weight = self.solve_output(feature_map, train_y)

            pred = self.forward(train_x)
            loss = self.loss_fn(pred, train_y).item()

            vpred= self.forward(val_x)
            vloss = self.loss_fn(vpred, val_y).item()

            self.logger.info('Hidden size: {} \t \nTraining MSE: {:.8f} \t Validating MSE: {:.8f} '.format(
                self.Channel_size, loss, vloss))

            save_checkpoint({
                'cv': self.params.cv,
                'weight_IH': self.Conv.weight.data,
                'bias_IH': self.Conv.bias.data,
                'weight_HO': self.out_weight}, checkpoint=self.params.model_dir)

    def predict(self, input, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        with torch.no_grad():
            input = torch.tensor(input).float().to(self.params.device)
            pred  = self.forward(input)
        return pred.cpu().numpy()