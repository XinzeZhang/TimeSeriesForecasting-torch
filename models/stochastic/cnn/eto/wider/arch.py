import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from re import S
# import math
import gc
from tqdm import trange,tqdm

import torch
import torch.nn as nn
import numpy as np

# from models.stochastic.Evolve import ReadoutTuner
from task.TaskLoader import Opt
# from models.stochastic.cnn.ice.wider.readoutTuner import ho_get
from models.stochastic.cnn.eto.wider.preTuner import PreTuner
from models.stochastic.cnn.eto.wider.subTuner import CellTrainer
from models.stochastic.cnn.eto.wider.regTuner import ReadoutTuner
from models.stochastic.cnn.eto.wider.basic import cell_ho, cal_eNorm

from models.stochastic.cnn.eto.wider.basic import wide_cell as cell
# from models.stochastic.cnn.ice.tuner_demo import Arch_dict, ArchTuner

from task.util import os_makedirs
# import importlib
import plotly.graph_objects as go
from  task.TaskLoader import torch_dataloader, dnn_dataset

# def patience_check(L):
#     return all(x<=y for x, y in zip(L, L[1:]))


class ETO_SDNN(nn.Module):
    """
    From WideNet ESN to Adaptive Convolutional ESN: An adaptive method to incrementally craft convolutional ESN for time series forecasting.
    """
    def __init__(self, opts, logger):
        super(ETO_SDNN, self).__init__()
        self.hyper = opts
        self.arch_choice = opts.arch_choice
        self.patience = opts.patience
        self.patience_bos = opts.patience_bos #the constructive process should early-stop after the WideNet > patience_bos and the valid-loss does not be smaller (aka. improve) than the valid-loss min(list[-patience:] )
        self.logger = logger
        self.device = self.hyper.device    
        self.Lag_order  = opts.lag_order
        self.Output_dim = opts.H
        self.Input_dim = opts.input_dim
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            
        if self.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.to(self.device)
            self.usingCUDA = True
        # self.branch_size = opts.branch_size # the largest branches of the finally neural architectures
        # self.best_size = 0
        # In the objection, the branch list should be a 2D tree, which stores the WideNet and the WideNet of the neural architecture.
        # branch: diversing branch
        # to clear
        # bough: main branch
        
        # during the first trial version, the max WideNet is set as 1, the arch. grows widther.
        # self.pointerTree = [] # to do, store the list of the pointers which point to the afore-arch from the current arch in the branch list.        
        self.mse = nn.MSELoss()
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        self.fit_info.l_E = []
        self.fit_info.p_E = []
        self.fit_info.c_E = []
        self.fit_info.r_E = []
        self.fit_info.r_lambda = []
        
        # self.WidthNetth = 0
        self.max_width = opts.max_cells
        self.best_width = None
        # self.WideNet = 0
        # self.max_width = 20
        self.init_arch()
    
    def patience_check(self):
        '''Check the validation metric convergence, if non-promotion, then early-stop'''
        pat_Tag = True
        curr_width = len(self.fit_info.vloss_list)
        if self.patience_bos <= self.patience:
            self.patience_bos = self.patience
        if curr_width > self.patience_bos:
            # min_vl = min(self.fit_info.vloss_list[-self.patience:])
            pat_min = min(self.fit_info.vloss_list[-self.patience:])
            his_min = min(self.fit_info.vloss_list)
            
            if his_min < pat_min:
                pat_Tag = False
            # pat_list = self.fit_info.vloss_list[-self.patience:]
            # avg_vl = sum(pat_list) / len(pat_list)
            # last_vl = self.fit_info.vloss_list[-1]
            # if last_vl > avg_vl:
            #     pat_Tag = False
        
        return pat_Tag
      
    def init_arch(self,):
        # WideNet: main branch; WideNet: extra branch
        # self.WideNet = nn.ModuleList([])
        self.WideNet = nn.ModuleList([])
            
    def loaderPack(self, tra_loader, val_loader):
        dataPack = Opt()
        dataPack.model_fit_dir = self.hyper.model_fit_dir
        # dataPack
        
        _sample = next(iter(tra_loader))
        dataPack.steps = _sample[0].shape[2]
        dataPack.input_dim = _sample[0].shape[1]
        dataPack.H = _sample[1].shape[1]
        dataPack.xm_size = dataPack.steps * dataPack.input_dim
        dataPack.device = self.device
            
        dataPack.x = []
        dataPack.y = []
        dataPack.e = []
        dataPack.cellcatP = []
        # init_p = []
        
        for _x, _y in (tra_loader):
            _x = _x.to(self.device)
            _y = _y.to(self.device)
            dataPack.x.append(_x)
            dataPack.y.append(_y)
            _e = _y.detach().clone()
            dataPack.e.append(_e)
        
        dataPack.catY = torch.cat(dataPack.y, dim=0)
        dataPack.catE = dataPack.catY.detach().clone()
        dataPack.catX = torch.cat(dataPack.x, dim=0)
        

        dataPack.catSumP = torch.zeros_like(dataPack.catY).to(dataPack.device)
        
        # self.norm_scale = dataPack.catE.size(0) * dataPack.H

        vx = []
        vy = [] 
        for _vx, _vy in val_loader:
            _vx = _vx.to(self.device)
            _vy = _vy.to(self.device)
            vx.append(_vx)
            vy.append(_vy)
            
        dataPack.catVX = torch.cat(vx, dim=0)
        # dataPack.catVY = torch.cat(vy, dim=0)
        dataPack.catVE =torch.cat(vy, dim=0)
        
        return dataPack
    
    
    def forward(self, input, widthNum = None):
        sum_pred = torch.zeros(
            (input.data.size(0), self.hyper.H)).to(self.device)
        _widthNum = widthNum if widthNum is not None else len(self.WideNet)
        for cell in self.WideNet[:_widthNum]:
            pred = cell(input) # ensure the net-arch is forward on WideNet
            sum_pred += pred
        return sum_pred
    
    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.hyper.batch_size if _batch_size is None else _batch_size
        set_data = dnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim)

        set_loader = torch_dataloader(set_data, batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader
    
    
    def xfit(self, train_data, val_data):
        '''
        '''
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        min_vrmse = float('inf')        
        dataPack = self.loaderPack(train_loader, val_loader)
            
        for Width_idx in trange(self.max_width):
            patience_pass = self.patience_check()
            if patience_pass == False:
                break
            
            # ToDo: try BOHB with combing PreTuner with cellTrainer
            preT = PreTuner(self.arch_choice)
            # to get the pretuning result: the best_config of the arch_delta, only the dataPack.X, dataPack.catX, dataPack.catE is needed.
            arch_config, best_arch, fc_io = preT.tuning(self.hyper, dataPack, len(self.WideNet), self.device)  # the e_norm returned by the arch_config is calculated on validation set
            
            # arch_delta = Opt(init=arch_config)
            
            
            _cell = cell(arch=best_arch, fc_io=fc_io)
            _cell.device = dataPack.device
            _cell.hyper = arch_config.hyper
            _cell.name = arch_config.name
            
            torch.cuda.empty_cache() if next(_cell.parameters()).is_cuda else gc.collect()
            
            
            
            # arch_delta.arch = best_arch
            # arch_delta.fc_io = fc_io
            # arch_delta.device = dataPack.device

            # cal_eNorm will give _cell a ho layer if it's None
            last_eNorm, curr_eNorm = cal_eNorm(_cell, dataPack) #these e_Norm are calculated on the training dataset
            
            
            _cell.last_eNorm = last_eNorm
            _cell.pT_eNorm = curr_eNorm
            
            pT_eNorm = curr_eNorm
            
            if 'cTrain' in self.hyper.dict: # remaining debug
                if self.hyper.cTrain is True:

                    
                    self.hyper.cTrain_info.WidthNum = len(self.WideNet)
                    self.hyper.cTrain_info.cid = self.hyper.cid
                    self.hyper.cTrain_info.sid = self.hyper.sid
                    
                    cellTrainer = CellTrainer(_cell, dataPack, self.hyper.cTrain_info)
                    cellTrainer.training()
                    _cell = cellTrainer._cell
                    
                    torch.cuda.empty_cache() if next(_cell.parameters()).is_cuda else gc.collect()
                    # after the training, _cell has the property of cT_eNorm
            
            if 'rTune' in self.hyper.dict: # remaining debug
                if self.hyper.rTune is True:

                    hoTuner = ReadoutTuner()
                    _cell = hoTuner.tuning(self.hyper, _cell, dataPack, len(self.WideNet))
                    
                    torch.cuda.empty_cache() if next(_cell.parameters()).is_cuda else gc.collect()
            
            
            self.logger.info(_cell)
            ho = cell_ho(_cell, dataPack)
            _cell.ho = ho
        
            self.WideNet.append(_cell)
            # dataPack = arch_delta.data
            torch.cuda.empty_cache() if next(_cell.parameters()).is_cuda else gc.collect()
            
            dataPack = self.data_Update(dataPack,_cell)
            
            rmse = np.sqrt(self.mse(dataPack.catSumP, dataPack.catY).item())
            self.fit_info.loss_list.append(rmse)
            
            vp = []
            vy = []
            
            for v_x, v_y in val_loader:
                v_x = v_x.to(self.device)
                v_y = v_y.to(self.device)
                vy.append(v_y)
                v_p = self.forward(v_x)
                vp.append(v_p)
            
            vp = torch.cat(vp, dim=0)
            vy = torch.cat(vy, dim=0)
            vrmse = np.sqrt(self.mse(vp, vy).item())
            self.fit_info.vloss_list.append(vrmse)
            
            if vrmse < min_vrmse:
                min_vrmse = vrmse
                self.best_width = Width_idx
                self.logger.info('Found new best state')
                self.logger.info('Best vrmse: {:.4f}'.format(min_vrmse))
            
            self.logger.info('WideNet size: {} \t Arch type: {} \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                Width_idx+1, _cell.name, rmse, vrmse, min_vrmse))
            
            self.logger.info("**** Curr Error 2-norm (before preTuning): {}".format(last_eNorm))
            self.fit_info.l_E.append(last_eNorm)
            self.logger.info("**** Curr Error 2-norm (after preTuning): {}".format(pT_eNorm))
            self.fit_info.p_E.append(pT_eNorm)
            
            if hasattr(_cell, 'cT_eNorm'):
                self.logger.info("**** Curr Error 2-norm (after cellTraining): {}".format(_cell.cT_eNorm))
                self.fit_info.c_E.append(_cell.cT_eNorm)
            if hasattr(_cell, 'rT_eNorm'):
                self.logger.info("**** Curr Error 2-norm (after hoTuning): {}".format(_cell.rT_eNorm))
                self.fit_info.r_E.append(_cell.rT_eNorm)
                self.fit_info.r_lambda.append(_cell.ho.lambda_reg)
                
            self.logger.info("**** Curr Error 2-norm (after all-stage): {}".format(torch.linalg.matrix_norm(dataPack.catE).item()))
            self.logger.info('>'*36)
            
        self.fit_info.trmse = self.fit_info.loss_list[self.best_width]
        self.fit_info.vrmse = min_vrmse
        # ToDo plot the e_norm lines
        return self.fit_info

    def eNorm2mse(self, _list):
        se = np.power(np.asarray(_list), 2)
        mse = se / self.norm_scale
        return mse

    def plot_eNorm(self, ):
        plot_dir = os.path.join(self.hyper.model_fit_dir, 'figures')
        os_makedirs(plot_dir)
        _x = np.linspace(0, 1, len(self.fit_info.loss_list))
        
        # transfer all metric to MSE
        loss_info = np.power(np.asarray(self.fit_info.loss_list), 2)
        vloss_info = np.power(np.asarray(self.fit_info.vloss_list), 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_x, y=loss_info, mode = 'lines', name = 'tra. mse'))
        fig.add_trace(go.Scatter(x=_x, y=vloss_info, mode = 'lines', name = 'val. mse'))
        lE = self.eNorm2mse(self.fit_info.l_E)
        pE = self.eNorm2mse(self.fit_info.p_E)
        
        fig.add_trace(go.Scatter(x=_x, y=lE, mode = 'markers', name = 'lT. mse'))
        fig.add_trace(go.Scatter(x=_x, y=pE, mode = 'markers', name = 'pT. mse'))
        
        if hasattr(self.hyper, 'cTraining'):
            cE = self.eNorm2mse(self.fit_info.c_E)
            fig.add_trace(go.Scatter(x=_x, y=cE, mode = 'markers', name = 'cT. mse'))
        
        if hasattr(self.hyper, 'rTuning'):
            rE = self.eNorm2mse(self.fit_info.r_E)
            fig.add_trace(go.Scatter(x=_x, y=rE, mode = 'markers', name = 'rT. mse'))
        
        fig.write_image(os.path.join(plot_dir, 'mse_lines.png'))
        
    def data_Update(self, dataPack, cell):
        
        catCurP = cell(torch.cat(dataPack.x, dim=0))
        
        dataPack.cellcatP.append(catCurP)
        
        # if cell_id == 0:
        #     for i, _x in enumerate(dataPack.x):
        #         _, _, pred = cell(_x)
        #         dataPack.cellP[cell_id][i] = pred
        # else:
        #     currentP = []
        #     for i, _x in enumerate(dataPack.x):
        #         _, _, pred = cell(_x)
        #         currentP.append(pred)
                
        #     dataPack.cellP.append(currentP)
            
        #     # ToDo: may reduce the memory usage by only saving the SumP list rather than every cellP
                
        # catCurP = torch.cat(dataPack.cellP[-1], dim=0)
        
        dataPack.catSumP += catCurP
        dataPack.catE = dataPack.catE - catCurP
        
        vcatCurP = cell(dataPack.catVX)
        dataPack.catVE = dataPack.catVE - vcatCurP
        
        total_samples = 0
        for i, _e in enumerate(dataPack.e):
            batch_size = _e.size(0)
            dataPack.e[i] = dataPack.catE[total_samples:(total_samples+batch_size),:]    
            total_samples += batch_size    
        
        # e_check = torch.allclose(dataPack.catE, torch.cat(dataPack.e, dim=0)) # the calculation of the dataPack.e has been checked
        
        return dataPack

    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.forward(batch_x, widthNum= self.best_width + 1)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred
    # def loss_fn(self, Hidden, Hidden_delta, E):
    #     '''To Do'''
    #     pass
    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred    