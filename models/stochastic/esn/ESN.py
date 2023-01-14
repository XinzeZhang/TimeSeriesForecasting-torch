import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from task.TaskLoader import Opt
from models.stochastic.Base import esnLayer
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from  task.TaskLoader import rnn_dataset, torch_dataloader
from task.gpu_mem_track import MemTracker
from tqdm import trange
from task.metric import rmse
import math

class EchoStateNetwork(nn.Module):
    """Basic Leaky Integrator Echo State Network (ESN)

    'Jaeger, Herbert, et al. "Optimization and applications of echo state networks with leaky-integrator neurons." Neural networks 20.3 (2007): 335-352.'

    Neuron reservoir with recurrent connection and random
    weights. Leaky factor (or damping) ensure echoes in the network. No
    learning takes place in the reservoir, readout is left at the user's
    convience. The input processed by these ESN should be normalized in [-1, 1]

    Parameters
    ----------
    Leaky rate : float, optional
        Leaky (forget) factor for echoes, strong impact on the dynamic of the
        reservoir. Possible values between 0 and 1, default is 1 which is equal to basic ESN.

    weight_scaling : float, optional
        Spectral radius of the reservoir, i.e. maximum eigenvalue of the weight
        matrix, also strong impact on the dynamical properties of the reservoir.
        Classical regimes involve values around 1, default is 0.9

    discard_steps : int, optional
        Discard first steps of the timeserie, to allow initialization of the
        network dynamics.
    """

    def __init__(self, opts=None, logger=None):
        super(EchoStateNetwork, self).__init__()
        self.opts = opts
        self.logger = logger

        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.Input_dim = opts.input_dim
        self.Lag_order  = opts.lag_order
        self.Time_steps = self.Lag_order - self.Input_dim + 1 

        self.Horizon = opts.H
    
        self.Output_dim = opts.H
        self.Hidden_Size = opts.hidden_size

        self.device = opts.device
        self._device = torch.device('cpu')

        assert opts.readout_steps > 0
        self.Readout_steps = -self.Time_steps if opts.readout_steps > self.Time_steps else -opts.readout_steps
        assert self.Readout_steps < 0


        self.fc_io = opts.fc_io
        if self.fc_io not in ['step', 'series', 'off']:
            raise ValueError(
                "Unknown fc_io type '{}'".format(self.fc_io)
            )

        self.init_arch()
        self.loss_fn = nn.MSELoss()

        if self.opts.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)
            self.usingCUDA = True
            # self.gpu_tracker = MemTracker()

        self.fit_info = Opt()

    def init_arch(self,):
        self.layer_esn = esnLayer(
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        self.layer_esn.freeze()

        if self.fc_io == 'step':
            self.readout_size = self.Hidden_Size + self.Input_dim
        elif self.fc_io == 'series':
            self.readout_size = self.Hidden_Size + self.Lag_order
        else:
            self.readout_size = self.Hidden_Size

        self.readout = nn.Linear(
            self.readout_size, self.Output_dim)
        self.readout.weight.requires_grad = False
        self.readout.bias.requires_grad = False

    def check_state_strip(self, length):
        state_select = False
        for i in range(length):
            if 'strip_{}'.format(i) in self.opts.dict:
                state_select = True
                break
        
        if state_select:
            for i in range(length):
                if 'strip_{}'.format(i) not in self.opts.dict:
                    raise ValueError('Missing hyper config: "strip_{}"!'.format(i))
        
        return state_select
                

    def init_state_strip(self,):
        state_select = self.check_state_strip(self.Time_steps)
        
        if state_select:
            self.strip_operator = [self.opts.dict['strip_{}'.format(i)] for i in range(self.Time_steps)]
        else:
            strip_operator = np.zeros(self.Time_steps)
            strip_operator[self.Readout_steps:] = 1
            self.strip_operator = strip_operator.tolist()
            
        assert len(self.strip_operator) == self.Time_steps

    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_loader = torch_dataloader(rnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim), batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader

    def gen_series_x(self, x, device):
        if self.fc_io == 'series':
            series_x = torch.empty(size=(x.shape[0], self.Lag_order)).to(device)
            series_x[:, :self.Time_steps] = x[:, 0, :self.Time_steps]
            if self.Input_dim > 1:
                series_x[:, self.Time_steps:] = x[:,1:, self.Time_steps -1]
        else:
            series_x = None

        return series_x

    def io_check(self, hidden, x):
        x = x.to(self._device)
        hidden = hidden.to(self._device)
        if self.fc_io == 'step':
            hidden = torch.cat(
                (x, hidden), dim=1)
        elif self.fc_io == 'series':
            series_x = self.gen_series_x(x, self._device)
            full_x = series_x.unsqueeze(2).repeat(1,1, hidden.shape[2])
            hidden = torch.cat((full_x, hidden), dim=1)
        
        return hidden

    def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self._device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (self.opts.lambda_reg * torch.eye(HTH.size(0))).to(self._device)
        A = HTH + I
        # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            W = torch.mm(torch.linalg.pinv(A.cpu()),
                         HTY.cpu()).t().to(self._device)
        return W, tag

    def update_readout(self, f_Hidden, x, y):
        f_Hidden = self.io_check(f_Hidden, x)
        
        self.init_state_strip()
        f_Hidden = self.strip_process(f_Hidden)
        y = self.strip_process(y)
        
        W, tag = self.solve_output(f_Hidden, y)
        self.readout.bias = nn.Parameter(W[:, 0], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        self.logger.info('Global LSM: {} \t L2 regular: {}'.format(
            tag, 'True' if self.opts.lambda_reg != 0 else 'False'))

    def strip_process(self, data):
        '''data shape: samples, dim, steps
        '''  
        select = []
        read_operator = self.strip_operator.copy()
    
        assert data.shape[2] == len(read_operator)
        
        for id, tag in enumerate(read_operator):
            if int(tag) == 1:
                select.append(data[:,:, id])
    
        if len(select) == 0:
            select = data[:,:, -1:]
        else:
            select = torch.stack(select, dim=2).to(self._device)

        select = select.permute(0, 2, 1)
        select = torch.flatten(select, start_dim=0, end_dim=1).to(self._device)
        return select
    
    def forward(self, x):
        hidden = self.reservior_transform(x)
        hidden = self.io_check(hidden, x)
        output = self.readout(hidden[:,:,-1])
        return output

    def reservior_transform(self, x):
        Hidden_States, _ = self.layer_esn(x)
        Hidden_States = Hidden_States.to(self._device)
        torch.cuda.empty_cache() if self.usingCUDA else gc.collect()
        return Hidden_States

    def batch_transform(self, data_loader, cat = True):
        h_states = []
        x = []
        y = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # _h_states = self.reservior_transform(batch_x)
            h_states.append(self.reservior_transform(batch_x).to(self._device))
            
            x.append(batch_x.to(self._device))
            y.append(batch_y.to(self._device))
            
            torch.cuda.empty_cache() if self.usingCUDA else gc.collect()
        
        if cat:  
            h_states = torch.cat(h_states, dim=0)
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
        
        return h_states, x, y

    def xfit(self, train_data, val_data):
        
        # min_vmse = 9999
        
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        h_states, x, y = self.batch_transform(train_loader)
        
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        self.update_readout(h_states, x, y)
        
        pred = self.readout(self.io_check(h_states, x)[:,:,-1])
        
        y = y[:,:,-1].cpu().numpy()
        pred = pred.cpu().numpy()
        self.fit_info.trmse = rmse(y,pred)
        # vh_states, val_x, val_y = self.batch_transform(val_loader)
        # vpred = self.readout(self.io_check(vh_states, val_x)[:,:,-1])
        _, val_y, vpred = self.loader_pred(val_loader)

        self.fit_info.vrmse = rmse(val_y, vpred)

        self.xfit_logger()

        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        return self.fit_info

    def xfit_logger(self,):
        # self.logger.info('Hidden size: {}'.format(self.Hidden_Size))
        self.logger.critical(
            'Training RMSE: {:.4g} \t Validating RMSE: {:.4g}'.format(self.fit_info.trmse, self.fit_info.vrmse))

    def predict(self, x):
        x = torch.tensor(x).to(torch.float32).to(self.device)
        output = self.forward(x)
        pred = output.detach().cpu().numpy()
        return pred
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y[:,:,-1]
            batch_pred = self.forward(batch_x)
            x.append(batch_x.cpu())
            y.append(batch_y.cpu())
            pred.append(batch_pred.cpu())
        x = torch.cat(x, dim=0).detach().numpy()
        y = torch.cat(y, dim=0).detach().numpy()
        pred = torch.cat(pred, dim=0).detach().numpy()
        
        return x, y, pred
    
    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred
    
class SSO_ESN(EchoStateNetwork):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)
        
        self.Horizon = opts.H
       
        self.Loss_steps = -(-self.Readout_steps + self.Horizon - 1)
        self.State_steps = self.Time_steps+ self.Horizon - 1
    
    def init_arch(self):
        self.Output_dim = 1
        self.encoder = esnLayer(
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        self.encoder.freeze()
        
        self.decoder = self.encoder
        self.decoder.freeze()
        
        if self.fc_io == 'step':
            self.readout_size = self.Hidden_Size + self.Input_dim
        elif self.fc_io == 'series':
            self.readout_size = self.Hidden_Size + self.Lag_order
        else:
            self.readout_size = self.Hidden_Size
        
        self.readout = nn.Linear(
            self.readout_size, self.Output_dim)
        self.readout.weight.requires_grad = False
        self.readout.bias.requires_grad = False

    def init_state_strip(self, ):
        state_select = self.check_state_strip(self.State_steps)

        for i in range(self.State_steps):
            if 'strip_{}'.format(i) not in self.opts.dict:
                state_select = False
                break
        
        if state_select:
            self.strip_operator = [self.opts.dict['strip_{}'.format(i)] for i in range(self.State_steps)]
        else:
            strip_operator = np.zeros(self.State_steps)
            strip_operator[self.Loss_steps:] = 1
            self.strip_operator = strip_operator.tolist()
            
        assert len(self.strip_operator) == self.State_steps

    def data_loader(self, data, _batch_size = None):
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_loader = torch_dataloader(rnn_dataset(data, self.Horizon, self.Lag_order,self.Input_dim, sso=True), batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader
            
    def step_io_cat(self, last_x, last_hidden, series_x= None, device= torch.device('cpu')):
        if self.fc_io == 'step':
            cat_hidden = torch.cat((last_x.to(device),last_hidden.to(device)), dim=1 )
        elif self.fc_io == 'series':
            cat_hidden = torch.cat((series_x.to(device), last_hidden.to(device)), dim= 1)
        else:
            cat_hidden = last_hidden.to(device)
            
        return cat_hidden

    def forward(self, x):
        output = torch.empty(size=(x.shape[0], self.Horizon)).to(self._device)
        
        _, last_hidden = self.encoder(x[:,:, :self.Time_steps])# need check
        # print(x[0,:,:self.Time_steps][:, -1])
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        
        self.logger.info('Start decoding...')
        
        series_x = self.gen_series_x(x, self._device)

        for j in trange(self.Horizon):
            # output[0,:]= torch.arange(1001,1001+self.Horizon).view(-1,) # only for testing
            if self.Input_dim - j > 0:
                last_x = x[:, :(self.Input_dim-j), self.Time_steps - 1 +j].to(self._device) 
                last_output = output[:,:j]
                last_x = torch.cat((last_x, last_output), dim=1)
            else:
                last_x = output[:,(j-self.Input_dim):j]

            cat_hidden = self.step_io_cat(last_x, last_hidden, series_x, self._device)
            output[:,j] = self.readout(cat_hidden).view(-1,)
            if j < self.Horizon -1:
                _, last_hidden = self.decoder(last_x.unsqueeze(dim=2).to(self.device), last_hidden.to(self.device))
            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()   
        return output
    
    
    def reservior_transform(self, x):
        '''
        pass unitTest
        '''
        # print(x[0,:,:self.Time_steps][:,-1])
        enc_states, last_state = self.encoder(x[:,:, :self.Time_steps])
        # enc_states = enc_states[:,:, self.Readout_steps:]
        enc_states = enc_states.to(self._device)
        torch.cuda.empty_cache() if self.usingCUDA else gc.collect()
        
        # print(x[0,:,self.Time_steps:][:,0])
        dec_states, _ = self.decoder(x[:,:,self.Time_steps:], last_state)
        dec_states = dec_states.to(self._device)
        torch.cuda.empty_cache() if self.usingCUDA else gc.collect()
        assert dec_states.shape[2] == self.Horizon -1 
        
        cat_states = torch.cat((enc_states, dec_states),dim=2)
        assert cat_states.shape[2] == self.Time_steps + self.Horizon - 1
        return cat_states

    def strip_process(self, data):
        '''data shape: samples, dim, steps
        '''  
        select = []
        read_operator = self.strip_operator.copy()
    
        assert data.shape[2] == len(read_operator)
        
        for id, tag in enumerate(read_operator):
            if int(tag) == 1:
                select.append(data[:,:, id])
    
        if len(select) == 0:
            select = data[:,:, -self.Horizon:]
        else:
            select = torch.stack(select, dim=2)
        
        select = select.permute(0, 2, 1)
        select = torch.flatten(select, start_dim=0, end_dim=1)
        return select
    
   
    def fit(self, train_loader):
        h_states, x, y = self.batch_transform(train_loader)
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        self.update_readout(h_states, x, y)
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
    
    def xfit(self, train_data, val_data):
        # min_vmse = 9999
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        with torch.no_grad():
            self.fit(train_loader)
            _, y, pred = self.loader_pred(train_loader)
            self.fit_info.trmse = rmse(y, pred)
    
            _, val_y, vpred = self.loader_pred(val_loader)
            self.fit_info.vrmse = rmse(val_y, vpred)
            
        # if math.isinf(self.fit_info.vrmse) :
        #     raise RuntimeError('Validation inf. error encountered.')
        #     self.grad_readout(train_loader, val_loader)
        # elif self.fit_info.vrmse > self.opts.tol:
        #     self.grad_readout(train_loader, val_loader)

        self.xfit_logger()
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        return self.fit_info

    # def grad_readout(self, train_loader, val_loader):
    #     self.readout.to(self.device)
    #     self.readout.weight.requires_grad = True
    #     self.readout.bias.requires_grad = True
    #     self.optimizer = torch.optim.Adam(self.readout.parameters(), lr=0.001)
    #     self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
    #         self.optimizer, 10, gamma=0.9)
        
    #     min_vmse = 9999
    #     train_len = len(train_loader)
        
    #     self.fit_info.loss_list = []
    #     self.fit_info.vloss_list = []
        
    #     self.epochs = 100
    #     for epoch in trange(self.epochs):
    #         self.logger.info(
    #             'Epoch {}/{}'.format(epoch + 1, self.epochs))
    #         mse_train = 0

    #         for i, (batch_x, batch_y) in enumerate(train_loader):
    #             batch_x = batch_x.to(torch.float32).to(self.device)
    #             batch_y = batch_y.to(torch.float32).to(self.device)
    #             self.optimizer.zero_grad()
                
    #             hidden = self.reservior_transform(batch_x)
    #             series_x = self.gen_series_x(batch_x, self.device)
    #             pred = torch.empty(size=(batch_x.shape[0], self.Horizon)).to(self.device)
                
    #             for j in range(self.Horizon):
    #                 cat_hidden = self.step_io_cat(batch_x[:,:, j - self.Horizon], hidden[:, :, j -self.Horizon], series_x=series_x, device=self.device)
    #                 pred[:,j] = self.readout(cat_hidden).view(-1,)
                
    #             batch_y = batch_y[:,:, - self.Horizon:].squeeze(1)
                
    #             # y_pred = y_pred.squeeze(1)
    #             loss = self.loss_fn(pred, batch_y)
    #             loss.backward()
    #             mse_train += loss.item()
    #             self.optimizer.step()
                
    #         mse_train = mse_train / train_len
    #         self.fit_info.loss_list.append(mse_train)
            
    #         self.epoch_scheduler.step()
            
    #         with torch.no_grad():
    #             self._device = self.device
    #             _, val_y, vpred = self.loader_pred(val_loader)
    #             mse_val = self.loss_fn(torch.tensor(vpred), torch.tensor(val_y)).item()

            
    #         self.fit_info.vloss_list.append(mse_val)
            
    #         self.logger.info('Current training loss: {:.4f} \t validating loss: {:.4f}'.format(mse_train,mse_val))
            
    #         if mse_val < min_vmse:
    #             min_vmse = mse_val
    #             self.logger.info('Found new best readout')
    #             self.logger.info('Best vmse: {:.4f}'.format(min_vmse))
    #             self.best_readout = self.readout.state_dict()

    #     self.readout.load_state_dict(self.best_readout) 
    #     with torch.no_grad():
    #         _, y, pred = self.loader_pred(train_loader)
    #         self.fit_info.trmse = rmse(y, pred)
    
    #         _, val_y, vpred = self.loader_pred(val_loader)
    #         self.fit_info.vrmse = rmse(val_y, vpred)
    
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_pred = self.forward(batch_x)
            assert batch_y.shape[1] == 1
            batch_y = batch_y.to(self._device)[:,:, -self.Horizon:].squeeze(1)
            x.append(batch_x.cpu())
            y.append(batch_y.cpu())
            pred.append(batch_pred.cpu())
        x = torch.cat(x, dim=0).detach().numpy()
        y = torch.cat(y, dim=0).detach().numpy()
        pred = torch.cat(pred, dim=0).detach().numpy()
        
        return x, y, pred  

    
class S2S_ESN(SSO_ESN):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)

    def init_arch(self):
        self.Output_dim = 1
        self.Readout_steps = -1
        
        self.encoder = esnLayer(
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        self.encoder.freeze()
        
        self.decoder = esnLayer(
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        self.decoder.freeze()
        
        if self.fc_io == 'step':
            self.readout_size = self.Hidden_Size + self.Input_dim
        elif self.fc_io == 'series':
            self.readout_size = self.Hidden_Size + self.Lag_order
        else:
            self.readout_size = self.Hidden_Size
        
        self.readout = nn.Linear(
            self.readout_size, self.Output_dim)
        self.readout.weight.requires_grad = False
        self.readout.bias.requires_grad = False        
        
    def init_state_strip(self,):

        strip_operator = np.zeros(self.State_steps)
        strip_operator[-self.Horizon:] = 1
        self.strip_operator = strip_operator.tolist()
            
        assert len(self.strip_operator) == self.State_steps        