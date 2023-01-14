import os,sys

from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import numpy as np
import torch.nn as nn
import torch
from tqdm import trange
import gc
from models.stochastic.Base import esnLayer
from models.stochastic.esn.ESN import EchoStateNetwork
import sys

from tqdm.std import tqdm


class Growing_ESN(EchoStateNetwork):
    """
    Growing Echo-State Network With Multiple Subreservoirs, TNNLS 2017
    """

    def __init__(self, opts=None, logger=None):
        # self.iw_bound = opts.iw_bound
        self.Subreservoir_Size = opts.branch_size  # 最多新增加的隐藏层单元的个数
        opts.fc_io = 'off'
        super().__init__(opts, logger)


        # assert self.lambda_reg == 0
        self.best_res_idx = self.Subreservoir_Size - 1
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        self.init_state_strip()

    def init_arch(self, ):
        self.res_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])

        for w in range(self.Subreservoir_Size):
            layer_esn = esnLayer(
                init='svd',
                hidden_size=self.Hidden_Size,
                input_dim=self.Input_dim,
                nonlinearity=self.opts.nonlinearity,
                leaky_r=self.opts.leaky_r,
                weight_scale=self.opts.weight_scaling,
                iw_bound=self.opts.iw_bound,
                hw_bound=self.opts.hw_bound,
                device=self.device
            )
            layer_esn.freeze()
            self.res_list.append(layer_esn)

            readout_size = (w+1)*self.Hidden_Size
            readout = nn.Linear(readout_size, self.Output_dim)
            self.ho_list.append(readout)

    def forward(self, x, res_index = None):
        if res_index is None:
            res_index = self.best_res_idx
        H_state = self.transform(x, res_index)
        output = self.ho_list[res_index](H_state)
        return output

    # 根据存储的▲ih和▲hh生成完整的H（Hidden_States）
    def transform(self, x, res_index):
        samples = x.shape[0]
        H_state = torch.empty(samples, self.Hidden_Size *
                              (res_index+1)).to(self._device)
        for i in range(res_index + 1):
            _, h_state = self.res_list[i](x)
            h_state = h_state.to(self._device)
            # h_state = self.reservior_transform(x, i)
            H_state[:, i *
                    self.Hidden_Size:(i+1)*self.Hidden_Size] = h_state
        return H_state

    #根据新增的▲ih和▲hh生成▲H（每一步的增量Hidden_States）
    # def reservior_transform(self, x, i):
    #     _, last_state = self.res_list[i](x)
    #     return last_state
    
    def reservior_transform(self, x, i):
        '''
        return: layer_hidden_state, _\n
        layer_hidden_state: (samples,Hidden_Size, time_steps)\n
        '''
        Hidden_States, _ = self.res_list[i](x)
        Hidden_States = Hidden_States.to(self._device)
        torch.cuda.empty_cache() if self.usingCUDA else gc.collect()
        return Hidden_States

    def update_readout(self, Hidden, x, y, i):
        
        Hidden = self.io_check(Hidden, x)
        Hidden = self.strip_process(Hidden)
        y = self.strip_process(y)
        W, tag = self.solve_output(Hidden, y)
        self.ho_list[i].bias = nn.Parameter(W[:, 0], requires_grad=False)
        self.ho_list[i].weight = nn.Parameter(W[:, 1:], requires_grad=False)
        self.logger.info('LSM: {} \t L2 regular: {}'.format(
            tag, 'True' if self.opts.lambda_reg != 0 else 'False'))

    def batch_transform(self, data_loader, res_index):
        h_states = []
        x = []
        y = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            _h_states = self.reservior_transform(batch_x, res_index)
            h_states.append(_h_states)
            x.append(batch_x)
            y.append(batch_y)
        h_states = torch.cat(h_states, dim=0)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return h_states, x, y

    def xfit(self, train_data, val_data):
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        min_vrmse = float('inf')

        H_state_list = []
        vH_state_list = []
        for i in trange(self.Subreservoir_Size):
            self.logger.info('-'*55)
            self.logger.info(
                'Subs index: {} \t Reservoir size: {}'.format(
                    i, self.Hidden_Size * (i+1)))

            #  solve the output
            h_state, x, y = self.batch_transform(train_loader, i)
            H_state_list.append(h_state)
            H_state = torch.cat(H_state_list, dim=1)

            self.update_readout(H_state, x, y, i)
            
            
            pred = self.ho_list[i](self.io_check(H_state, x)[:,:,-1])
            loss = np.sqrt(self.loss_fn(pred, y[:,:,-1].cpu()).item())
            self.fit_info.loss_list.append(loss)

            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

            v_h_state, val_x, val_y = self.batch_transform(val_loader, i)
            vH_state_list.append(v_h_state)
            vH_state = torch.cat(vH_state_list, dim=1)

            vpred = self.ho_list[i](self.io_check(vH_state, val_x)[:,:,-1])
            vloss = np.sqrt(self.loss_fn(vpred, val_y[:,:,-1].cpu()).item())
            self.fit_info.vloss_list.append(vloss)

            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

            if vloss < min_vrmse:
                min_vrmse = vloss
                self.best_res_idx = i
                self.logger.info('****** Found new best state ******')
                # self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))
            self.logger.info('Best VRMSE: {:.8f} \t Best Res_idx: {} \t  Training RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
                min_vrmse, self.best_res_idx, loss, vloss))

        self.fit_info.trmse = self.fit_info.loss_list[self.best_res_idx]
        self.fit_info.vrmse = min_vrmse

        return self.fit_info

    def predict(self, x):
        x = torch.tensor(x).to(torch.float32).to(self.opts.device)
        output = self.forward(x, self.best_res_idx)
        pred = output.detach().cpu().numpy()
        return pred




# class Incremental_ESN(Growing_ESN):
#     """
#     Incrementally configure the output weight
#     """

#     def __init__(self, opts=None, logger=None):
#         super().__init__(opts, logger)

#     def init_arch(self, ):
#         self.res_list = nn.ModuleList([])
#         self.ho_list = nn.ModuleList([])

#         for w in range(self.Subreservoir_Size):
#             layer_esn = esnLayer(
#                 init='svd',
#                 hidden_size=self.Hidden_Size,
#                 input_dim=self.Input_dim,
#                 nonlinearity=self.opts.nonlinearity,
#                 leaky_r=self.opts.leaky_r,
#                 weight_scale=self.opts.weight_scaling,
#                 iw_bound=self.opts.iw_bound,
#                 hw_bound=self.opts.hw_bound,
#                 device=self.device
#             )
#             layer_esn.freeze()
#             self.res_list.append(layer_esn)

#             readout_size = self.Hidden_Size
#             if self.fc_io:
#                 readout_size += self.Input_dim * \
#                     (self.Time_steps - self.Discard_steps)
#             readout = nn.Linear(readout_size, self.Output_dim)
#             self.ho_list.append(readout)
#             ## Attention!!! There is another manner of implementation, which
#             # fc_io is only calculated once and independent to all sub-reservoirs.

#     def forward(self, x, res_index = None):
#         if res_index is None:
#             res_index = self.best_res_idx
        
#         sum_pred = torch.zeros(x.data.size(
#             0), self.Output_dim).to(self.opts.device)

#         #  subs belongs to [1, branch_size]
#         for i in range(res_index + 1):
#             # Hidden_State = Hidden_States[:, sub * self.Hidden_Size:(sub + 1) * self.Hidden_Size]
#             h_state = self.reservior_transform(x, i)
#             h_state = self.io_check(h_state, x)
#             pred = self.ho_list[i](h_state)
#             sum_pred += pred

#         return sum_pred

#     def xfit(self, train_loader, val_loader):
#         min_vrmse = float('inf')
#         y = []
#         for _, batch_y in tqdm(train_loader):
#             batch_y = batch_y.to(self.device)
#             y.append(batch_y)
#         y = torch.cat(y, dim=0)
#         Error = y.detach().clone()
#         pred = torch.zeros_like(y).to(self.device)

#         val_y = []
#         for _, batch_y in tqdm(val_loader):
#             batch_y = batch_y.to(self.device)
#             val_y.append(batch_y)
#         val_y = torch.cat(val_y, dim=0)
#         vpred = torch.zeros_like(val_y).to(self.device)

#         for i in trange(self.Subreservoir_Size):
#             subs = i + 1
#             self.logger.info('-'*55)
#             self.logger.info(
#                 'Subs size: {} \t Reservoir size: {}'.format(
#                     subs, self.Hidden_Size * subs))

#             h_state, x, y = self.batch_transform(train_loader, i)
#             self.update_readout(h_state, x, Error, i)
#             pred = pred + self.ho_list[i](self.io_check(h_state, x))
#             #增量更新方式
#             #训练
#             Error = y - pred
#             #存储已经得到的预测值
#             loss = np.sqrt(self.loss_fn(pred, y).item())
#             self.fit_info.loss_list.append(loss)

#             vh_state, val_x, val_y = self.batch_transform(val_loader, i)
#             vpred = vpred + self.ho_list[i](self.io_check(vh_state, val_x))

#             vloss = np.sqrt(self.loss_fn(vpred, val_y).item())
#             self.fit_info.vloss_list.append(vloss)

#             if vloss < min_vrmse:
#                 min_vrmse = vloss
#                 self.best_res_idx = i
#                 self.logger.info('****** Found new best state ******')
#                 # self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))
#             self.logger.info('Best VRMSE: {:.8f} \t Best Res_idx: {} \t  Training RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
#                 min_vrmse, self.best_res_idx, loss, vloss))

#         self.fit_info.trmse = self.fit_info.loss_list[self.best_res_idx]
#         self.fit_info.vrmse = min_vrmse

#         return self.fit_info
