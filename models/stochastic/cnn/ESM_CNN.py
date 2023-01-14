import os
import sys
from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from tqdm.std import tqdm
from task.TaskLoader import Opt
from models.stochastic.Base import cnnLayer
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import gc
from  task.TaskLoader import torch_dataloader, dnn_dataset
import math


class ESM_CNN(nn.Module):
    # Incremental CNN
    def __init__(self, opts=None, logger=None):
        super(ESM_CNN, self).__init__()
        self.opts = opts
        self.logger = logger
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.Lag_order  = opts.lag_order
        self.Output_dim = opts.H
        self.Input_dim = opts.input_dim
        self.Time_steps = self.Lag_order - self.Input_dim + 1         

        self.Candidate_size = opts.candidate_size if 'candidate_size' in opts.dict else 10
        self.Channel_size = opts.channel_size  if 'channel_size' in opts.dict else 10
        self.Kernel_size = opts.kernel_size if 'kernel_size' in opts.dict else math.ceil(self.Lag_order / 4)
        
        self.Kernel_size = max(2, self.Kernel_size)
        self.Kernel_list = opts.Kernel_list if 'Kernel_list' in opts.dict else list(dict.fromkeys([math.ceil(
            self.Time_steps / 3), math.ceil(self.Time_steps / 4), math.ceil(self.Time_steps / 5), math.ceil(self.Time_steps / 6)]))
        
        self.Kernel_list = [max(k, 2) for k in self.Kernel_list ]
                
        self.p_size = opts.p_size if 'p_size' in opts.dict else 3
        self.device = opts.device

        self.hw_lambda = opts.hw_lambda if 'hw_lambda' in opts.dict else 0.5
        
        self.tolerance = opts.tolerance if 'tolerance' in opts.dict else 0
        self.lambda_reg = opts.lambda_reg if 'lambda_reg' in opts.dict else 0
        
        self.nonlinearity = opts.nonlinearity if 'nonlinearity' in opts.dict else 'tanh'
        
        self.init_arch()
        self.loss_fn = nn.MSELoss()

        self.best_conv_id = opts.channel_size - 1
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        if self.opts.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)
            self.usingCUDA = True

    def init_arch(self,):
        self.conv_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])

    # @profile
    def forward(self, input, conv_idx = None):
        if conv_idx is None:
            conv_idx = self.best_conv_id
        # conv_idx : channel num
        sum_pred = torch.zeros(
            (input.data.size(0), self.Output_dim)).to(self.opts.device)
        # if not using_best:
        for i in range(conv_idx + 1):
            feature_map = self.channel_transform(
                input, i)
            pred = self.ho_list[i](feature_map)
            sum_pred += pred

        return sum_pred

    # @profile
    def channel_transform(self, input, i, conv_weight=None, conv_bias=None):
        '''
        Using list to storage the channel-weight for multiple kernel-size filters
        '''
        if conv_weight is not None and conv_bias is not None:
            k_size = conv_weight.data.size(2)
            Conv = nn.Conv1d(self.Input_dim, 1, k_size,
                             padding=0).to(self.opts.device)
            Conv.weight.data = conv_weight
            Conv.bias.data = conv_bias

            feature_map = Conv(input)
            feature_map = torch.sigmoid(feature_map)
            feature_map = self.Pool(feature_map)
            feature_map = feature_map.view(
                feature_map.data.size(0), feature_map.data.size(2))
        else:
            _, feature_map = self.conv_list[i](input)

        return feature_map

    # @profile
    def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self.device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (0 * torch.eye(HTH.size(0))).to(self.device)
        A = HTH + I
        # A = A.to(self.device)
        # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
        orig_rank = torch.linalg.matrix_rank(A.cpu(), hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            W = torch.mm(torch.linalg.pinv(A.cpu()),
                         HTY.cpu()).t().to(self.device)
        return W, tag

    def update_readout(self, Hidden,  y, show = False):
        # _Hidden = self.io_check(Hidden, x)
        W, tag = self.solve_output(Hidden, y)
        if show:
            self.logger.info('LSM: {} \t L2 regular: {}'.format(            tag, 'True' if self.lambda_reg != 0 else 'False'))
        ho = nn.Linear(Hidden.size(1), y.size(1))
        ho.bias = nn.Parameter(W[:, 0], requires_grad=False)
        ho.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        ho.inverse = tag
        return ho

    def filter_search(self, dataloader, error):
        # for Lambda in self.Lambdas:
        best_kernel = Opt()
        best_kernel.loss = float('inf')

        for kernel_size in self.Kernel_list:
            conv_sets = cnnLayer(
                input_dim=self.Input_dim,
                out_channels=self.Candidate_size,
                kernel_size=kernel_size,
                hw_bound=(-self.hw_lambda, self.hw_lambda),
                nonlinearity=self.nonlinearity,
                device=self.device
            )
            conv_sets.freeze()
            conv_fms = self.batch_transform(dataloader, conv_sets)
            # size(N, Channel_out, Steps_out)
            for i, idx in enumerate(range(self.Candidate_size)):
                c_idx = conv_fms[:, idx, :]
                ho = self.update_readout(c_idx, error)
                pred = ho(c_idx)
                loss = np.sqrt(self.loss_fn(pred, error).item())

                if loss < best_kernel.loss:
                    best_kernel.loss = loss
                    best_kernel.kernel_size = kernel_size
                    best_kernel.conv_weight = conv_sets.conv.weight[i, :, :].reshape(
                        1, self.Input_dim, kernel_size).detach().clone()
                    best_kernel.conv_bias = conv_sets.conv.bias[i].view(
                        -1).detach().clone()
                    best_kernel.ho = ho

                torch.cuda.empty_cache() if next(conv_sets.parameters()).is_cuda else gc.collect()
                
        channel = cnnLayer(
            input_dim=self.Input_dim,
            out_channels=1,
            kernel_size=best_kernel.kernel_size,
            nonlinearity=self.nonlinearity,
            device=self.device
        )
        channel.update(best_kernel.conv_weight, best_kernel.conv_bias)
        channel.freeze()
        channel.kernel_size = best_kernel.kernel_size
        return channel, best_kernel.ho

    def batch_transform(self, data_loader, channel):
        h_states = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            fm, _ = channel(batch_x)
            h_states.append(fm)
                        
            torch.cuda.empty_cache() if next(channel.parameters()).is_cuda else gc.collect()
                        
        h_states = torch.cat(h_states, dim=0)
        torch.cuda.empty_cache() if next(channel.parameters()).is_cuda else gc.collect()
        return h_states

        
    # @profile
    def xfit(self, train_data, val_data):
        # min_vmse = 9999
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        min_vrmse = float('inf')
        x = []
        y = []
        for batch_x, batch_y in (train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            x.append(batch_x)
            y.append(batch_y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        Error = y.detach().clone()
        pred = torch.zeros_like(y).to(self.device)

        val_y = []
        for _, batch_y in (val_loader):
            batch_y = batch_y.to(self.device)
            val_y.append(batch_y)
        val_y = torch.cat(val_y, dim=0)
        v_pred = torch.zeros_like(val_y).to(self.device)

        for i in trange(self.Channel_size):
            channel, ho = None, None

            channel, ho = self.filter_search(train_loader, Error)
            
            self.logger.info('LSM: {} \t L2 regular: {}'.format(            ho.inverse, 'True' if self.lambda_reg != 0 else 'False'))
            
            self.conv_list.append(channel)
            self.ho_list.append(ho)


            # update error
            h_states = self.batch_transform(train_loader, channel)
            pred = pred + self.ho_list[i](torch.flatten(h_states, start_dim=1))
            Error = y - pred

            loss = np.sqrt(self.loss_fn(pred, y).item())
            self.fit_info.loss_list.append(loss)

            vh_state = self.batch_transform(val_loader, channel)
            v_pred = v_pred + self.ho_list[i](torch.flatten(vh_state, start_dim=1))
            vloss = np.sqrt(self.loss_fn(v_pred, val_y).item())
            self.fit_info.vloss_list.append(vloss)

            # vmse = vloss
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
            if vloss < min_vrmse:
                min_vrmse = vloss
                self.best_conv_id = i
                self.logger.info('Found new best state')
                self.logger.info('Best vrmse: {:.4f}'.format(min_vrmse))

            self.logger.info('Channel size: {} \t Kernel size: {} \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                i+1, channel.kernel_size, loss, vloss, min_vrmse))

        self.fit_info.trmse = self.fit_info.loss_list[self.best_conv_id]
        self.fit_info.vrmse = min_vrmse

        return self.fit_info

    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_data = dnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim)
        # set_if_x_data = if_choose(set_data.data,self.if_operator)
        # set_data.data = set_if_x_data
        
        set_loader = torch_dataloader(set_data, batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader

    def predict(self, x,):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        x = torch.tensor(x).float().to(self.opts.device)
        output = self.forward(x, self.best_conv_id)
        pred = output.detach().cpu().numpy()
        return pred

    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.forward(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred

    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred
    
class ES_CNN(ESM_CNN):
    def __init__(self, opts=None, logger=None):
        nn.Module.__init__(self,)
        # super().__init__(opts, logger)
        self.opts = opts
        self.logger = logger
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.Lag_order  = opts.lag_order
        self.Input_dim = opts.input_dim if 'input_dim' in opts.dict else 1
        self.Time_steps = self.Lag_order - self.Input_dim + 1    
        
        self.Output_dim = opts.H
        self.Channel_size = opts.channel_size  if 'channel_size' in opts.dict else 10
        self.Kernel_size = opts.kernel_size if 'kernel_size' in opts.dict else math.ceil(self.Lag_order / 4)
        self.Kernel_size = max(2, self.Kernel_size)
        
        self.p_size = opts.p_size if 'p_size' in opts.dict else 3
        self.device = opts.device

        self.hw_lambda = opts.hw_lambda if 'hw_lambda' in opts.dict else 0.5
        
        self.tolerance = opts.tolerance if 'tolerance' in opts.dict else 0
        self.lambda_reg = opts.lambda_reg if 'lambda_reg' in opts.dict else 0
        
        self.nonlinearity = opts.nonlinearity if 'nonlinearity' in opts.dict else 'tanh'

        self.init_arch()
        self.loss_fn = nn.MSELoss()

        self.best_conv_id = opts.channel_size - 1
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        if self.opts.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)
            self.usingCUDA = True


    def init_arch(self, ):
        self.conv_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])

        for w in range(self.Channel_size):
            channel = cnnLayer(
                input_dim=self.Input_dim,
                out_channels=1,
                kernel_size=self.Kernel_size,
                hw_bound=(-self.hw_lambda, self.hw_lambda),
                pooling_size=self.p_size,
                nonlinearity=self.nonlinearity,
                device=self.device
            )
            channel.freeze()
            self.conv_list.append(channel)

            # readout_size = self.Time_steps - self.Kernel_size - self.p_size + 2
            # readout = nn.Linear(readout_size, self.Output_dim)
            # self.ho_list.append(readout)

    # @profile
    def xfit(self, train_data, val_data):
        # min_vmse = 9999
        self.init_input_strip()
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        min_vrmse = float('inf')
        x = []
        y = []
        for batch_x, batch_y in (train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            x.append(batch_x)
            y.append(batch_y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        Error = y.detach().clone()
        pred = torch.zeros_like(y).to(self.device)

        val_y = []
        for _, batch_y in tqdm(val_loader):
            batch_y = batch_y.to(self.device)
            val_y.append(batch_y)
        val_y = torch.cat(val_y, dim=0)
        v_pred = torch.zeros_like(val_y).to(self.device)

        for i in trange(self.Channel_size):
            # update error
            h_states = self.batch_transform(train_loader, self.conv_list[i])
            h_states = torch.flatten(h_states, start_dim=1)
            self.ho_list.append(self.update_readout(h_states, Error)) 
            
            self.logger.info('LSM: {} \t L2 regular: {}'.format(            self.ho_list[i].inverse, 'True' if self.lambda_reg != 0 else 'False'))
                            
            pred = pred + self.ho_list[i](h_states)
            Error = y - pred

            loss = np.sqrt(self.loss_fn(pred, y).item())
            self.fit_info.loss_list.append(loss)

            vh_state = self.batch_transform(val_loader, self.conv_list[i])
            v_pred = v_pred + self.ho_list[i](torch.flatten(vh_state, start_dim=1))
            vloss = np.sqrt(self.loss_fn(v_pred, val_y).item())
            self.fit_info.vloss_list.append(vloss)

            # vmse = vloss
            # self.logger.info('Current vmse: {:.4f}'.format(vmse))
            if vloss < min_vrmse:
                min_vrmse = vloss
                self.best_conv_id = i
                self.logger.info('Found new best state')
                self.logger.info('Best vrmse: {:.4f}'.format(min_vrmse))

            self.logger.info('Channel size: {} \t Kernel size: {} \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                i+1, self.Kernel_size, loss, vloss, min_vrmse))

        self.fit_info.trmse = self.fit_info.loss_list[self.best_conv_id]
        self.fit_info.vrmse = min_vrmse

        return self.fit_info


class Stoc_CNN(ESM_CNN):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)

    def init_arch(self,):
        self.channel = cnnLayer(
            input_dim=self.Input_dim,
            out_channels=self.Channel_size,
            kernel_size=self.Kernel_size,
            hw_bound=(-self.hw_lambda, self.hw_lambda),
            pooling_size=self.p_size,
            device=self.device
        )
        self.channel.freeze()
        readout_size = self.Channel_size * \
            (self.Time_steps - self.Kernel_size - self.p_size + 2)
        self.readout = nn.Linear(readout_size, self.Output_dim)

    def forward(self, input):
        # input = input.permute(0, 2, 1)
        _, fm_view = self.channel(input)
        pred = self.readout(fm_view)
        return pred

    def xfit(self, train_data, val_data):
        # min_vmse = 9999
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        y = []
        for batch_x, batch_y in (train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            y.append(batch_y)
        y = torch.cat(y, dim=0)

        val_y = []
        for _, batch_y in tqdm(val_loader):
            batch_y = batch_y.to(self.device)
            val_y.append(batch_y)
        val_y = torch.cat(val_y, dim=0)

        h_states = self.batch_transform(train_loader, self.channel)
        self.readout = self.update_readout(h_states, y)
        pred = self.readout(h_states)
        loss = np.sqrt(self.loss_fn(pred, y).item())

        vh_states = self.batch_transform(val_loader, self.channel)
        v_pred = self.readout(vh_states)
        vloss = np.sqrt(self.loss_fn(v_pred, val_y).item())

        self.fit_info.trmse = loss
        self.fit_info.vrmse = vloss

        return self.fit_info

    def predict(self, input):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        with torch.no_grad():
            input = torch.tensor(input).float().to(self.opts.device)
            pred = self.forward(input)
        return pred.cpu().numpy()