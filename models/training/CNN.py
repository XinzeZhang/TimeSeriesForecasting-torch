import torch
import torch.nn as nn
# import torch.optim as optim
import numpy as np
from tqdm.std import tqdm
# from numpy import concatenate
# import torch.nn.init as Init
# from task._data_process import mape

from task.util import savebest_checkpoint, plot_all_epoch, plot_xfit,load_checkpoint
from task.util import os_makedirs,set_logger

import os
from tqdm import trange
import logging


class CNN(nn.Module):
    def __init__(self, params=None, logger = None):
        super(CNN, self).__init__()
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.input_dim = 1 + params.cov_dim
        self.Timesteps = params.steps
        self.Output_dim = params.H
        # self.Output_dim = output_dim
        self.Channel_size = params.channel_size
        self.Kernel_size = params.kernel_size
        self.p_size = 3

        self.params.plot_dir = os.path.join(self.params.task_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)
        
        self.arch_init()

        self.best_state = None


    def arch_init(self,):
        self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,self.Kernel_size,padding=0)
        # self.Conv.weight.data = Init.normal_(torch.empty(channel_size, in_channels, kernel_size).float(), std=0.015)
        # self.Conv.bias.data =  Init.normal_(torch.empty(channel_size).float(), std=0.015)
        self.fc = nn.Linear(self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2) ,self.Output_dim,bias=False)
        self.Pool = nn.AvgPool1d(kernel_size=self.p_size,stride=1,padding=0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.params.step_lr, gamma=0.9)
        
        self.epochs = self.params.epochs


        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)


    def forward(self,input):
        # input = input.permute(0, 2, 1)
        feature_map = self.Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        
        feature_map = feature_map.view(-1, self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2))
        
        pred = self.fc(feature_map)

        return pred
    
    def xfit(self, train_loader, val_loader, restore_file = None, checkpoint= False):
        
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.epochs))
        loss_avg = np.zeros((self.epochs))
        vloss_avg = np.zeros_like(loss_avg)

        for epoch in trange(self.epochs):
            self.logger.info(
                'Epoch {}/{}'.format(epoch + 1, self.epochs))
            mse_train = 0
            loss_epoch = np.zeros(train_len)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                mse_train += loss.item()
                loss_epoch[i] = loss.item()
                self.optimizer.step()

            mse_train = mse_train / train_len
            loss_summary[epoch * train_len:(epoch + 1) * train_len] = loss_epoch
            loss_avg[epoch] = mse_train

            self.epoch_scheduler.step()

            with torch.no_grad():
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(torch.float32).to(self.params.device)
                    batch_y = batch_y.to(torch.float32).to(self.params.device)
                    output = self(batch_x)
                    mse_val += self.loss_fn(output,
                                         batch_y).item()
                mse_val = mse_val / len(val_loader)
            vloss_avg[epoch] = mse_val

            self.logger.info('Current training loss: {:.4f} \t validating loss: {:.4f}'.format(mse_train,mse_val))
            vmse = mse_val
            self.logger.info('Current vmse: {:.4f}'.format(vmse))
            
            if vmse < min_vmse:
                min_vmse = vmse
                self.logger.info('Found new best state')
                best_epoch = epoch
                best_state = self.state_dict()
                best_optim = self.optimizer.state_dict()
                self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

        if checkpoint:
            savebest_checkpoint({
                    'epoch': best_epoch,
                    'cv': self.params.cv,
                    'state_dict': best_state,
                    'optim_dict': best_optim}, checkpoint=self.params.task_dir)
            self.logger.info(
                        'Checkpoint saved to {}'.format(self.params.task_dir))
        else:
            self.best_state = best_state

        # plot_all_epoch(loss_summary[:(
        #     epoch + 1) * train_len], self.params.dataset + '_cv{}_loss'.format(self.params.cv), self.params.plot_dir)
        plot_xfit(loss_avg,vloss_avg,self.params.dataset + '_cv{}_loss'.format(self.params.cv), self.params.plot_dir)

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.task_dir, 'best.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth) and using_best:
            self.logger.info('Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self, self.optimizer)
            
        elif using_best:
            self.load_state_dict(self.best_state)
        
        x = torch.tensor(x).to(torch.float32).to(self.params.device)
        output = self(x)
        pred = output.detach().cpu().numpy()

        return pred



class GS_CNN(nn.Module):
    def __init__(self, params=None, logger = None):
        super().__init__()
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

        self.params.plot_dir = os.path.join(self.params.task_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)
        
        self.params.task_dir = os.path.join(self.params.task_dir, 'gs')
        
    def gs_search(self, train_loader, val_loader):
        logger = logging.getLogger('gsCNN.gs')
        set_logger(os.path.join(self.params.task_dir, 'gs.log'), logger)
        channel_sizes = np.arange(10, 100, 10).tolist()
        best_i = 9999
        best_size = None
        for i, size in enumerate(tqdm(channel_sizes)):
            self.params.cv = i
            size= int(size)
            self.Channel_size = size
            self.arch_init()
            logger.info(
            'Loading the datasets and model for {}th-grid-search'.format(i))
            logger.info('Loading complete.')
            logger.info(f'Model: \n{str(self)}')
        
            i_v = self.xfit(train_loader, val_loader)
            
            if best_i >= i_v:
                best_i = i_v
                best_size = size

            logger.info("Channel Size: {}\tValidation Loss: {}".format(size, i_v))
            
            logger.info("Best Channel Size: {}".format(best_size))


    def arch_init(self,):
        self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,self.Kernel_size,padding=0)
        # self.Conv.weight.data = Init.normal_(torch.empty(channel_size, in_channels, kernel_size).float(), std=0.015)
        # self.Conv.bias.data =  Init.normal_(torch.empty(channel_size).float(), std=0.015)
        self.fc = nn.Linear(self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2) ,self.Output_dim,bias=False)
        self.Pool = nn.AvgPool1d(kernel_size=self.p_size,stride=1,padding=0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.params.step_lr, gamma=0.9)
        
        self.epochs = self.params.epochs


        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)



    def forward(self,input):
        # input = input.permute(0, 2, 1)
        feature_map = self.Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        
        feature_map = feature_map.view(-1, self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2))
        
        pred = self.fc(feature_map)

        return pred
    
    def xfit(self, train_loader, val_loader, restore_file = None):
        
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.epochs))
        loss_avg = np.zeros((self.epochs))
        vloss_avg = np.zeros_like(loss_avg)

        for epoch in trange(self.epochs):
            self.logger.info(
                'Epoch {}/{}'.format(epoch + 1, self.epochs))
            mse_train = 0
            loss_epoch = np.zeros(train_len)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                mse_train += loss.item()
                loss_epoch[i] = loss.item()
                self.optimizer.step()

            mse_train = mse_train / train_len
            loss_summary[epoch * train_len:(epoch + 1) * train_len] = loss_epoch
            loss_avg[epoch] = mse_train

            self.epoch_scheduler.step()

            with torch.no_grad():
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(torch.float32).to(self.params.device)
                    batch_y = batch_y.to(torch.float32).to(self.params.device)
                    output = self(batch_x)
                    mse_val += self.loss_fn(output,
                                         batch_y).item()
                mse_val = mse_val / len(val_loader)
            vloss_avg[epoch] = mse_val

            self.logger.info('Current training loss: {:.4f} \t validating loss: {:.4f}'.format(mse_train,mse_val))
            vmse = mse_val
            self.logger.info('Current vmse: {:.4f}'.format(vmse))
            if vmse < min_vmse:
                min_vmse = vmse
                self.logger.info('Found new best state')
                # savebest_checkpoint({
                #     'epoch': epoch,
                #     'cv': self.params.cv,
                #     'state_dict': self.state_dict(),
                #     'optim_dict': self.optimizer.state_dict()}, checkpoint=self.params.task_dir)
                # self.logger.info(
                #     'Checkpoint saved to {}'.format(self.params.task_dir))                        
                # self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

        plot_all_epoch(loss_summary[:(
            epoch + 1) * train_len], self.params.dataset + '_loss', self.params.plot_dir)
        plot_xfit(loss_avg,vloss_avg,self.params.dataset + '_loss', self.params.plot_dir)

        return min_vmse

    # def predict(self, x, using_best=True):
    #     '''
    #     x: (numpy.narray) shape: [sample, full-len, dim]
    #     return: (numpy.narray) shape: [sample, prediction-len]
    #     '''
    #     # test_batch: shape: [full-len, sample, dim]
    #     best_pth = os.path.join(self.params.task_dir, 'best.cv{}.pth.tar'.format(self.params.cv))
    #     if os.path.exists(best_pth) and using_best:
    #         self.logger.info('Restoring best parameters from {}'.format(best_pth))
    #         load_checkpoint(best_pth, self, self.optimizer)

    #     x = torch.tensor(x).to(torch.float32).to(self.params.device)
    #     output = self(x)
    #     pred = output.detach().cpu().numpy()

    #     return pred