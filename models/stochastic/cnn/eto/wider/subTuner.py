import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


import torch
import torch.nn as nn

from tqdm.std import trange
from task.TaskLoader import Opt

from models.stochastic.cnn.eto.wider.basic import close_ho, cell_ho, cal_eNorm, fcLayer,io_check

from task.util import set_logger, plot_xfit
# from task.util import plot_xfit
import copy
import numpy as np



mse_loss = nn.MSELoss()
class CellTrainer(Opt):
    def __init__(self, _cell, dataPack, info):
        super().__init__()
        
        self._cell = _cell
        self.dataPack = dataPack
        
        self.cid = info.cid
        self.sid = info.sid
        self.wNum = info.WidthNum
        
        self.max_epoch = info.max_epoch
        
        self.log_dir = ''
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        # self.loss_list = []
        # self.vloss_list = []
        self.device = dataPack.device
        self.training_name = info.training_name
        self.metric_loss = info.metric_loss

        
        self.last_state = Opt()
        self.last_state.arch = copy.deepcopy(self._cell.arch.state_dict())
        self.last_state.ho = copy.deepcopy(self._cell.ho.state_dict())
        
        self.best_state = Opt()
        self.best_state.arch = copy.deepcopy(self._cell.ho.state_dict())
        self.best_state.ho = copy.deepcopy(self._cell.ho.state_dict())
                
        self.batch_training = False
        
        self.host_dir  = os.path.join(dataPack.model_fit_dir, 'trainer.{}'.format(self.training_name), 'cv{}'.format(self.cid))
        
        cur_info = 'Wide{}.Arch{}'.format(info.WidthNum, self._cell.name.upper())
        
        self.model_state_path = os.path.join(self.host_dir, cur_info+'.pt')
        
        if self.wNum == 0:
            self.logger = set_logger(os.path.join(self.host_dir, 'cTrain.log'),'cTrainer.ser{}.cv{}.{}'.format(self.sid,self.cid,cur_info),level=20)
        else:
            self.logger = set_logger(os.path.join(self.host_dir, 'cTrain.log'),'cTrainer.ser{}.cv{}.{}'.format(self.sid,self.cid,cur_info),level=20,rewrite=False)
        
        for (arg, value) in self.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
        
        self.best_loss = float('inf')
        self.best_epoch = 0
   
    def training(self,):
        if os.path.exists(self.model_state_path):
            self._cell.load_state_dict(torch.load(self.model_state_path))
        else:
            try:
                if self.training_name == 'naive':
                    self.sample_tune()

                    self.load_best()
                
                plot_xfit(self.fit_info, 'cTraining_loss.Wide{}'.format(self.wNum), location=self.host_dir)
            except:
                self.logger.exception('!!!! Fail on naive training. Re-use the originally initialized arch. !!!!')
                self.load_last()
            
            torch.save(self._cell.state_dict(), self.model_state_path)

        self._cell.arch.freeze()
        self._cell.ho.freeze()
        
        _, cT_eNorm = cal_eNorm(self._cell, self.dataPack)
        self._cell.cT_eNorm = cT_eNorm
        
        self.logger.info('Best Error 2-norm (after cellTraining): {}\t Epoch: {}'.format(self.best_loss, self.best_epoch))
        self.logger.info('Current Error 2-norm (before preTuning): {}'.format(self._cell.last_eNorm))
        self.logger.info('Current Error 2-norm (after preTuning): {}'.format(self._cell.pT_eNorm))
        self.logger.info('Current Error 2-norm (after cellTraining): {}'.format(cT_eNorm))
        
        if next(self._cell.parameters()).is_cuda:
            torch.cuda.empty_cache()
    
    def sample_tune(self, report = True):
        self.logger.info('*'*50)
        self.logger.info('-------Start fine training-------')
        assert self._cell.ho is not None
        
        tag = 'Naive'
        with torch.no_grad():
            for epoch in trange(self.max_epoch):
                # torch.autograd.set_detect_anomaly(True)
                if next(self._cell.parameters()).is_cuda:
                    torch.cuda.empty_cache()
                
                if epoch > 0:
                    self._cell.arch.init_weights()
                    
                # weights_list = self._cell.arch.get_weights()
                # print(weights_list[0])
                
                # batch training
                tra_loss = 0    
                train_len = len(self.dataPack.e)
                for _x, _e in zip(self.dataPack.x, self.dataPack.e):
                    pred = self._cell(_x)
                    _loss = mse_loss(pred, _e)
                    tra_loss += _loss.item()
                tra_loss = tra_loss / train_len

                vpred = self._cell(self.dataPack.catVX)
                val_loss = mse_loss(vpred, self.dataPack.catVE)
                val_loss =val_loss.item()
            
                self.record_arch(tra_loss, val_loss, epoch, tag)
        
                if report:
                    self.logger.info('{} Training epoch: {} \t Training loss: {:.4e} \t Validation loss: {:.4e}'.format(tag, epoch, tra_loss, val_loss))
                self.fit_info.loss_list.append(tra_loss)
                self.fit_info.vloss_list.append(val_loss)
    
    def load_best(self,):
        self._cell.arch.load_state_dict(self.best_state.arch)
        self._cell.ho.load_state_dict(self.last_state.ho)
    
    def load_last(self,):
        self._cell.arch.load_state_dict(self.last_state.arch)
        self._cell.ho.load_state_dict(self.last_state.ho)
    
    def record_arch(self, tra_loss,val_loss, epoch, tag):
        if self.metric_loss == 'training':
            record_loss = tra_loss
        elif self.metric_loss == 'validation':
            record_loss = val_loss
        else:
            raise ValueError('Undefined metric_loss: {}'.format(self.metric_loss))
        
        if record_loss < self.best_loss:
            self.best_loss = record_loss
            self.best_state.arch = copy.deepcopy(self._cell.arch.state_dict())
            self.best_epoch = epoch
            
            if self.best_epoch != 0:
                self.logger.info('@@@@@@ Find new best cell state. @@@@@@')
                self.logger.info('{} Training epoch: {} \t Training loss: {:.4e} \t Validation loss: {:.4e}'.format(tag,epoch, tra_loss, val_loss))
