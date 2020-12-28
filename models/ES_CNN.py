import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.nn as nn
# import torch.nn.MSELoss as MSE
import torch.nn.init as Init

import numpy as np
import itertools

import gc
from tqdm import trange
from data_process.util import os_makedirs
from data_process.util import plot_xfit

def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    if model.params.device == torch.device('cpu'):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint, map_location='cuda')

    model.channels = checkpoint['channels']
    model.best_valid_channels = checkpoint['best_valid_channels']
    model.conv_weight = checkpoint['conv_weight']
    model.conv_bias = checkpoint['conv_bias']
    model.out_weight = checkpoint['out_weight']
    # model.loss_list = checkpoint['loss_list']
    # model.vloss_list = checkpoint['vloss_list']


def save_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(
        checkpoint, 'train.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)


class ES_CNN(nn.Module):
    def __init__(self, params=None, logger=None):
        super(ES_CNN, self).__init__()
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.In_channels = params.In_channels
        self.Timesteps = params.steps
        self.Output_dim = params.H
        self.Channel_size = params.channel_size
        self.Candidate_size = params.candidate_size
        self.Kernel_size = params.kernel_size
        self.Kernel_list = params.kernel_list
        self.p_size = 3


        self.Lambdas = params.Lambdas
        self.Lambdas_std = params.Lambdas_std
        self.r = params.r
        self.tolerance = params.tolerance


        with torch.no_grad():
            self.init_convWeight = Init.uniform_(torch.empty(
                1, self.In_channels, self.Kernel_size).float(), a=-0.25, b=0.25).to(self.params.device)
            self.init_convBias = Init.uniform_(
                torch.empty(1).float(), a=-0.25, b=0.25).to(self.params.device)
        self.channels = 1
        self.best_valid_channels = 1

        self.conv_weight = []
        self.conv_weight.append(self.init_convWeight)
        self.conv_bias = []
        self.conv_bias.append(self.init_convBias)

        self.out_weight = []

        # self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,self.Kernel_size,padding=0)
        self.Pool = nn.AvgPool1d(kernel_size=self.p_size, stride=1, padding=0)
        # self.fc  = nn.Linear(self.Channel_size * (self.Timesteps - self.Kernel_size -self.p_size +2) ,self.Output_dim,bias=False)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []
        self.vloss_list = []

        self.params.plot_dir = os.path.join(params.model_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

    # @profile
    def forward(self, input, channels):
        sum_pred = torch.zeros(
            (input.data.size(0), self.Output_dim)).to(self.params.device)
        # if not using_best:
        for channel in range(channels):
            feature_map = self.channel_transform(
                input, self.conv_weight[channel], self.conv_bias[channel])
            pred = self.fc_transform(feature_map, channel)
            sum_pred += pred
        # del feature_map, pred
        # gc.collect()
        return sum_pred

    # @profile
    def channel_transform(self, input, conv_weight, conv_bias):
        '''
        Using list to storage the conv-weight for multiple kernel-size filters
        '''
        k_size = conv_weight.data.size(2)
        Conv = nn.Conv1d(self.In_channels, 1, k_size,
                         padding=0).to(self.params.device)
        Conv.weight.data = conv_weight
        Conv.bias.data = conv_bias

        feature_map = Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        feature_map = feature_map.view(
            feature_map.data.size(0), feature_map.data.size(2))

        # del Conv
        # gc.collect()
        return feature_map

    # @profile
    def solve_output(self, feature, target):
        with torch.no_grad():
            C_state = torch.cat(
                (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
            output_w, _ = torch.lstsq(target, C_state)
            output_w = output_w[0:C_state.size(1)].to(self.params.device)
        return output_w

    # @profile
    def fc_transform(self, feature, epoch):
        C_state = torch.cat(
            (torch.ones(feature.size(0), 1).to(self.params.device), feature), 1)
        pred = C_state.mm(self.out_weight[epoch])
        # del C_state
        # gc.collect()
        return pred


    # @profile
    def xfit(self, train_loader, val_loader, restore_file=None):
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

            if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
                self.logger.info(
                    'Restoring parameters from {}'.format(restore_file))
                load_checkpoint(restore_file, self)
            else:
                # calcaulate the initial fit error with the first channel
                C_state_X = self.channel_transform(
                    train_x, self.init_convWeight, self.init_convBias)
                output_w = self.solve_output(C_state_X, train_y.detach().clone())
                self.out_weight.append(output_w)

            loss = 9999
            for i in trange(self.Channel_size):
                if i == self.channels:
                    success = True
                    Lambda = self.Lambdas[0]
                    new_conv_weight = torch.empty(
                        1, self.In_channels, self.Kernel_size).uniform_(-Lambda, Lambda).float().to(self.params.device)

                    new_conv_bias = torch.empty(
                        1).uniform_(-Lambda, Lambda).float().to(self.params.device)

                    feature_map = self.channel_transform(train_x, new_conv_weight, new_conv_bias)
                    new_output_w = self.solve_output(feature_map, error)

                    if not success or loss < self.tolerance:
                        break
                    else:
                        self.conv_weight.append(new_conv_weight)
                        self.conv_bias.append(new_conv_bias)
                        self.out_weight.append(new_output_w)
                        self.channels += 1

                # update error
                pred = self.forward(train_x, i+1)
                error = train_y - pred
                loss = self.loss_fn(pred, train_y).item()
                self.loss_list.append(loss)

                vpred = self.forward(val_x, i+1)
                vloss = self.loss_fn(vpred, val_y).item()
                self.vloss_list.append(vloss)


                # vmse = vloss
                # self.logger.info('Current vmse: {:.4f}'.format(vmse))
                if vloss < min_vmse:
                    min_vmse = vloss
                    self.best_valid_channels = self.channels
                    self.logger.info('Found new best state')
                    self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

                self.logger.info('Channel size: {} \t Kernel size: {} \nTraining MSE: {:.8f} \t Validating MSE: {:.8f} \t Best VMSE: {:.8f}'.format(
                    i+1, self.conv_weight[i].data.size(2), loss, vloss, min_vmse))
                save_checkpoint({
                    'channels': self.channels,
                    'best_valid_channels': self.best_valid_channels,
                    'cv': self.params.cv,
                    'conv_weight': self.conv_weight,
                    'conv_bias': self.conv_bias,
                    'out_weight': self.out_weight,
                    'loss_list': self.loss_list,
                    'vloss_list': self.vloss_list}, checkpoint=self.params.model_dir)

                self.logger.info(
                    'Checkpoint saved to {}'.format(self.params.model_dir))

                gc.collect()

            plot_xfit(np.array(self.loss_list), np.array(self.vloss_list),
                      self.params.dataset + '_cv{}_loss'.format(self.params.cv), self.params.plot_dir)

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.model_dir,
                                'train.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth) and using_best:
            self.logger.info(
                'Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self)
        with torch.no_grad():
            x = torch.tensor(x).to(torch.float32).to(self.params.device)
            output = self.forward(x, self.best_valid_channels)
            pred = output.detach().cpu().numpy()

        return pred

        



