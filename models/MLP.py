import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from data_process.util import savebest_checkpoint, load_checkpoint,plot_all_epoch,plot_xfit, os_makedirs

import numpy as np

from sklearn.metrics import mean_squared_error

from tqdm import trange
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.optim as optim



class MLP(nn.Module):
    def __init__(self, params=None, logger=None):
        super().__init__()
        self.params = params
        self.logger = logger

        timesteps, output_dim, hidden_size =  params.steps, params.H, params.hidden_size

        self.Input_dim = timesteps
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size

        self.num_epochs = self.params.num_epochs

        self.hidden = nn.Linear(self.Input_dim,self.Hidden_Size)
        self.fc = nn.Linear(self.Hidden_Size,self.Output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, params.step_lr, gamma=0.9)
        self.loss_fn = nn.MSELoss()

        self.params.plot_dir = os.path.join(params.model_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)


    def forward(self, input):
        input = input[:,:,0]
        h=self.hidden(input)
        h=torch.sigmoid(h)
        pred =self.fc(h)
        return pred

        self.Input_dim = input_dim
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size

        self.Num_iters = num_iters
        self.Optim_method = optim_method
        self.Learn_rate = learning_rate
        self.plot_ = plot_
        self.device = device

        self.verbose = True


    def xfit(self, train_loader, val_loader, restore_file=None):
        # update self.params
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.num_epochs))
        loss_avg = np.zeros((self.num_epochs))
        vloss_avg = np.zeros_like(loss_avg)

        epoch = 0
        for epoch in trange(self.num_epochs):
            # self.logger.info(
            #     'Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            mse_train = 0
            loss_epoch = np.zeros(train_len)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                # y_pred = y_pred.squeeze(1)
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
                    # output = output.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(batch_y.detach().cpu().numpy())
                    mse_val += self.loss_fn(output,
                                         batch_y).item()
                mse_val = mse_val / len(val_loader)
            vloss_avg[epoch] = mse_val

            preds = np.concatenate(preds)
            true = np.concatenate(true)

            # self.logger.info('Current training loss: {:.4f} \t validating loss: {:.4f}'.format(mse_train,mse_val))
            
            vmse = mean_squared_error(true, preds)
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
            if vmse < min_vmse:
                min_vmse = vmse
                # self.logger.info('Found new best state')
                savebest_checkpoint({
                    'epoch': epoch,
                    'cv': self.params.cv,
                    'state_dict': self.state_dict(),
                    'optim_dict': self.optimizer.state_dict()}, checkpoint=self.params.model_dir)
                # self.logger.info(
                #     'Checkpoint saved to {}'.format(self.params.model_dir))                        
                # self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

        plot_all_epoch(loss_summary[:(
            epoch + 1) * train_len], self.params.dataset + '_loss_cv{}'.format(self.params.cv), self.params.plot_dir)
        plot_xfit(loss_avg,vloss_avg,self.params.dataset + '_loss_cv{}'.format(self.params.cv), self.params.plot_dir)

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.model_dir, 'best.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth) and using_best:
            # self.logger.info('Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self, self.optimizer)

        x = torch.tensor(x).to(torch.float32).to(self.params.device)
        output = self(x)
        # output = output.squeeze(1)
        pred = output.detach().cpu().numpy()

        return pred