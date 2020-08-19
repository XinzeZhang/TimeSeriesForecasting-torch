from data_process.util import savebest_checkpoint, load_checkpoint,plot_all_epoch,plot_xfit
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import trange
import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))




class ConvRNN(nn.Module):
    def __init__(self, params=None, logger=None):
        super().__init__()
        self.params = params
        self.logger = logger

        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        input_dim, timesteps, output_dim = 1 + params.cov_dim, params.steps, params.H
        kernel_size1, kernel_size2, kernel_size3 = 7, 5, 3
        n_channels1, n_channels2, n_channels3 = params.n_channels1, params.n_channels2, params.n_channels3
        n_units1, n_units2, n_units3 = params.n_units1, params.n_units2, params.n_units3

        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.avg_pool2 = nn.AvgPool1d(4, 4)
        # input_dim denotes input_channel and is going to be transformed to nn_channels1, output_step = input_step - (kernel_size -1), where the input_step has been left padded with kernel_size - 1, thus the output_step should be equal to original timesteps
        self.conv11 = nn.Conv1d(input_dim, n_channels1,
                                kernel_size=kernel_size1)
        # channel transformation, samely outputs the same timesteps with left padding.
        self.conv12 = nn.Conv1d(n_channels1, n_channels1,
                                kernel_size=kernel_size1)
        self.conv21 = nn.Conv1d(input_dim, n_channels2,
                                kernel_size=kernel_size2)
        self.conv22 = nn.Conv1d(n_channels2, n_channels2,
                                kernel_size=kernel_size2)
        self.conv31 = nn.Conv1d(input_dim, n_channels3,
                                kernel_size=kernel_size3)
        self.conv32 = nn.Conv1d(n_channels3, n_channels3,
                                kernel_size=kernel_size3)

        # using the output_channel (n_channels1) as the input_size in each time steps, and using n_units1 to denote the hidden_size
        self.gru1 = nn.GRU(n_channels1, n_units1, batch_first=True)
        self.gru2 = nn.GRU(n_channels2, n_units2, batch_first=True)
        self.gru3 = nn.GRU(n_channels3, n_units3, batch_first=True)
        self.linear1 = nn.Linear(n_units1+n_units2+n_units3, output_dim)
        self.linear2 = nn.Linear(input_dim*timesteps, output_dim)
        # padding left (kernel_size1-1) step with value 0, padding right 0 step with value 0
        self.zp11 = nn.ConstantPad1d(((kernel_size1-1), 0), 0)
        self.zp12 = nn.ConstantPad1d(((kernel_size1-1), 0), 0)
        self.zp21 = nn.ConstantPad1d(((kernel_size2-1), 0), 0)
        self.zp22 = nn.ConstantPad1d(((kernel_size2-1), 0), 0)
        self.zp31 = nn.ConstantPad1d(((kernel_size3-1), 0), 0)
        self.zp32 = nn.ConstantPad1d(((kernel_size3-1), 0), 0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, params.step_lr, gamma=0.9)
        self.loss_fn = nn.MSELoss()


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # line1
        y1 = self.zp11(x)
        y1 = torch.relu(self.conv11(y1))
        y1 = self.zp12(y1)
        y1 = torch.relu(self.conv12(y1))
        y1 = y1.permute(0, 2, 1)
        out, h1 = self.gru1(y1)
        # line2
        y2 = self.avg_pool1(x)
        y2 = self.zp21(y2)
        y2 = torch.relu(self.conv21(y2))
        y2 = self.zp22(y2)
        y2 = torch.relu(self.conv22(y2))
        y2 = y2.permute(0, 2, 1)
        out, h2 = self.gru2(y2)
        # line3
        y3 = self.avg_pool2(x)
        y3 = self.zp31(y3)
        y3 = torch.relu(self.conv31(y3))
        y3 = self.zp32(y3)
        y3 = torch.relu(self.conv32(y3))
        y3 = y3.permute(0, 2, 1)
        out, h3 = self.gru3(y3)
        h = torch.cat([h1[-1], h2[-1], h3[-1]], dim=1)
        out1 = self.linear1(h)
        out2 = self.linear2(x.contiguous().view(x.shape[0], -1))
        out = out1 + out2
        return out

    def xfit(self, train_loader, val_loader, restore_file=None):
        # update self.params
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.params.num_epochs))
        loss_avg = np.zeros((self.params.num_epochs))
        vloss_avg = np.zeros_like(loss_avg)

        for epoch in trange(self.params.num_epochs):
            self.logger.info(
                'Epoch {}/{}'.format(epoch + 1, self.params.num_epochs))
            mse_train = 0
            loss_epoch = np.zeros(train_len)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                y_pred = y_pred.squeeze(1)
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
                    output = output.squeeze(1)
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
                savebest_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'optim_dict': self.optimizer.state_dict()}, checkpoint=self.params.model_dir)
                self.logger.info(
                    'Checkpoint saved to {}'.format(self.params.model_dir))                        
                self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

        plot_all_epoch(loss_summary[:(
            epoch + 1) * train_len], self.params.dataset + '_loss', self.params.plot_dir)
        plot_xfit(loss_avg,vloss_avg,self.params.dataset + '_loss', self.params.plot_dir)
            
    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.model_dir, 'best.pth.tar')
        if os.path.exists(best_pth) and using_best:
            self.logger.info('Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self, self.optimizer)

        x = torch.tensor(x).to(torch.float32).to(self.params.device)
        output = self(x)
        output = output.squeeze(1)
        pred = output.detach().cpu().numpy()

        return pred
