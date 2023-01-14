import os
import sys

from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.util import savebest_checkpoint, load_checkpoint,plot_all_epoch,plot_xfit, os_makedirs
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import trange
import torch.nn as nn
import torch

from torch.nn.utils.rnn import PackedSequence

class ConvRNN(nn.Module):
    def __init__(self, params=None, logger=None):
        super().__init__()
        self.params = params
        self.logger = logger

        if self.params.log_level == 3:
            for (arg, value) in params.dict.items():
                self.logger.info("Argument %s: %r", arg, value)

        input_dim, timesteps, output_dim = 1 + params.cov_dim, params.steps, params.H

        self.conv1 = nn.Conv1d(input_dim, params.channel_size // 2, kernel_size = params.kernel_size)
        self.conv2 = nn.Conv1d(params.channel_size // 2, params.channel_size, kernel_size= params.kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.lstm = nn.LSTM(params.channel_size, params.hidden_size, batch_first= True)
        self.fc = nn.Linear(params.hidden_size, params.output_size )
        self.output = nn.Linear(params.output_size, output_dim)

    
        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, params.step_lr, gamma=0.9)
        self.loss_fn = nn.MSELoss()

        self.epochs = self.params.epochs

        self.params.plot_dir = os.path.join(params.task_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

        self.best_state = None


    def initHidden(self, input):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            batchSize = int(input.batch_sizes[0])
        else:
            batchSize = input.data.size(0)
        result = torch.empty(1,
                            batchSize, self.params.hidden_size).float().to(self.params.device)
        result = nn.init.normal_(result, std=0.015)
        return result


    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # line1
        fm = self.conv1(x)
        fm = self.relu(fm)
        fm = self.conv2(fm)
        fm = self.relu(fm)
        fm = self.pool(fm)
        # fm : (batch, feature, seq)

        fm = fm.permute(0,2,1)
        # fm should be (batch, seq, feature) to be the input of lstm
        h_state = self.initHidden(fm)
        c_state = self.initHidden(fm)


        lstm_out, (h_state_n, c_state_n) = self.lstm(fm, (h_state, c_state))

        fm = self.fc(h_state_n[0,:,:])
        fm = self.relu(fm)
        out = self.output(fm)

        return out

    def xfit_epoch(self, train_loader, val_loader, epoch):
        self.logger.info(
        'Epoch {}/{}'.format(epoch + 1, self.epochs))
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(torch.float32).to(self.params.device)
            batch_y = batch_y.to(torch.float32).to(self.params.device)
            self.optimizer.zero_grad()
            y_pred = self(batch_x)
            loss = self.loss_fn(y_pred, batch_y)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            # mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                output = self(batch_x)
                # output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
            #     mse_val += self.loss_fn(output,
            #                             batch_y).item()
            # mse_val = mse_val / len(val_loader)
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            
            vmse = mean_squared_error(true, preds)
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
        return np.sqrt(vmse)


    def xfit(self, train_loader, val_loader, restore_file=None, checkpoint =False):
        # update self.params
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.epochs))
        loss_avg = np.zeros((self.epochs))
        vloss_avg = np.zeros_like(loss_avg)

        epoch = 0
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

            self.logger.info('Current training loss: {:.4f} \t validating loss: {:.4f}'.format(mse_train,mse_val))
            
            vmse = mean_squared_error(true, preds)
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
            # epoch + 1) * train_len], self.params.dataset + '_cv{}_loss'.format(self.params.cv), self.params.plot_dir)
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
        # output = output.squeeze(1)
        pred = output.detach().cpu().numpy()

        return pred
