import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.TaskLoader import Opt

import numpy as np

from sklearn.metrics import mean_squared_error

from tqdm import trange, tqdm
import torch
import torch.nn as nn
# from torch.nn.utils.rnn import PackedSequence
# import torch.optim as optim



class MLP(nn.Module):
    def __init__(self, opts=None, logger=None):
        super().__init__()
        self.opts = opts
        self.logger = logger

        timesteps, output_dim, hidden_size =  opts.steps, opts.H, opts.hidden_size

        self.Input_dim = timesteps
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size

        self.epochs = self.opts.epochs

        self.hidden = nn.Linear(self.Input_dim,self.Hidden_Size)
        self.fc = nn.Linear(self.Hidden_Size,self.Output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opts.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, opts.step_lr, gamma=0.9)
        self.loss_fn = nn.MSELoss()


        # create missing directories


        if self.opts.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)

        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        
    def forward(self, input):
        # input = input[:,:,0]
        h=self.hidden(input)
        h=torch.sigmoid(h)
        pred =self.fc(h)
        return pred


    def xfit_epoch(self, train_loader, val_loader, epoch):
        '''
        #ToDO transfer to new version
        '''
        self.logger.info(
        'Epoch {}/{}'.format(epoch + 1, self.epochs))
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(torch.float32).to(self.opts.device)
            batch_y = batch_y.to(torch.float32).to(self.opts.device)
            self.optimizer.zero_grad()
            y_pred = self(batch_x)
            loss = self.loss_fn(y_pred, batch_y)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            # rmse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.opts.device)
                batch_y = batch_y.to(torch.float32).to(self.opts.device)
                output = self(batch_x)
                # output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
            #     rmse_val += self.loss_fn(output,
            #                             batch_y).item()
            # rmse_val = rmse_val / len(val_loader)
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            
            vmse = mean_squared_error(true, preds)
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
        return np.sqrt(vmse)


    def xfit(self, train_loader, val_loader):
        # update self.opts

        min_vrmse = 9999
        train_len = len(train_loader)

        epoch = 0
        for epoch in trange(self.epochs):
            # self.logger.info(
            #     'Epoch {}/{}'.format(epoch + 1, self.epochs))
            rmse_train = 0
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.opts.device)
                batch_y = batch_y.to(torch.float32).to(self.opts.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                # y_pred = y_pred.squeeze(1)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                rmse_train += np.sqrt(loss.item())

                self.optimizer.step()
            
            rmse_train = rmse_train / train_len

            self.fit_info.loss_list.append(rmse_train)
            
            self.epoch_scheduler.step()
            
            with torch.no_grad():
                rmse_val = 0
                # preds = []
                # true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(torch.float32).to(self.opts.device)
                    batch_y = batch_y.to(torch.float32).to(self.opts.device)
                    output = self(batch_x)
                    # output = output.squeeze(1)
                    # preds.append(output.detach().cpu().numpy())
                    # true.append(batch_y.detach().cpu().numpy())
                    rmse_val += np.sqrt(self.loss_fn(output,
                                         batch_y).item())
                rmse_val = rmse_val / len(val_loader)


            self.fit_info.vloss_list.append(rmse_val)
            
            
            # vmse = mean_squared_error(true, preds)
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
            if rmse_val < min_vrmse:
                min_vrmse = rmse_val
                # self.logger.info('Found new best state')
                self.best_epoch = epoch
                # If you only plan to keep the best performing model (according to the acquired validation loss), don’t forget that best_model_state = model.state_dict() returns a reference to the state and not its copy! You must serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) otherwise your best best_model_state will keep getting updated by the subsequent training iterations. As a result, the final model state will be the state of the overfitted model.
                
                self.best_state = self.state_dict()
                #     'Checkpoint saved to {}'.format(self.opts.task_dir))                        
                # self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))

            self.fit_info.trmse = rmse_train
            self.fit_info.vrmse = min_vrmse
            self.xfit_logger(epoch)
        
        return self.fit_info

    def xfit_logger(self, epoch):
        self.logger.info('Epoch: {}'.format(epoch))
        self.logger.critical(
            'Training RMSE: {:.4g} \t Validating RMSE: {:.4g}'.format(self.fit_info.trmse, self.fit_info.vrmse))

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        if using_best:
            self.load_state_dict(self.best_state)

        x = torch.tensor(x).to(torch.float32).to(self.opts.device)
        output = self(x)
        # output = output.squeeze(1)
        pred = output.detach().cpu().numpy()

        return pred
    
    def loader_pred(self, data_loader, using_best=True):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.predict(batch_x, using_best=using_best)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred