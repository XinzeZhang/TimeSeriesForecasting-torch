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
        self.best_state.arch = None
        self.best_state.ho = None
                
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

    def set_optim(self, all = False):
        self.learning_rate = 0.005
        self.step_lr = 5
        
        # self._cell.arch.active()
        self.optimizer = torch.optim.AdamW(self._cell.parameters() if all else self._cell.arch.parameters(), lr=self.learning_rate)   
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.step_lr, gamma=0.8)
   
    def training(self,):
        if os.path.exists(self.model_state_path):
            self._cell.load_state_dict(torch.load(self.model_state_path))
        else:
            try:
                if self.training_name == 'fine':
                    self.set_optim(all=True) # reset optim with training hidden_arch and ho_arch simultaneously.
                    self.fine_train()
                elif self.training_name == 'naive':
                    self.set_optim()
                    self.fine_train(ho_active=False)
                
                if self.best_epoch == 0:
                    self.logger.info('!!!! Non effective training, Re-use the originally initialized arch. !!!!')
                    self.load_last()
                else:
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
        
         
        
    def fine_train(self, ho_active = True, report = True):
        '''
        Todo, using close_ho and preTuning result to init the _cell, and training the _cell with mse, after that, update the readout with LS.
        '''
        # catX = 
        
        # assert self._cell.ho is not None
        self.logger.info('*'*50)
        self.logger.info('-------Start fine training-------')

        assert self._cell.ho is not None
        
        # self._cell.arch.active()
        # if ho_active:
        #     self._cell.ho.active()
        # else:
        #     self._cell.ho.freeze()
        # mse_loss = nn.MSELoss()
        tag = 'Fine' if ho_active else 'Naive'
        
        for epoch in trange(self.max_epoch):
            # torch.autograd.set_detect_anomaly(True)
            if next(self._cell.parameters()).is_cuda:
                torch.cuda.empty_cache()
            if self.batch_training:
            # batch training
                tra_loss = 0    
                train_len = len(self.dataPack.e)
                for _x, _e in zip(self.dataPack.x, self.dataPack.e):
                    self.optimizer.zero_grad()
                    pred = self._cell(_x)
                    _loss = mse_loss(pred, _e)
                    _loss.backward()
                    self.optimizer.step()
                    tra_loss += _loss.item()
                    
                tra_loss = tra_loss / train_len
                
                with torch.no_grad():
                    vpred = self._cell(self.dataPack.catVX)
                    val_loss = mse_loss(vpred, self.dataPack.catVE)
                    val_loss =val_loss.item()
                
                self.record_arch(tra_loss, val_loss, epoch, tag)
            else:
            # # whole training
                try:
                    self.optimizer.zero_grad()
                    # self._cell.arch.requires_grad_(True)
                    # self._cell.ho.requires_grad_(False)
                    self._cell.arch.active()
                    self._cell.ho.freeze()
 
                    pred = self._cell(self.dataPack.catX)
                    _loss = mse_loss(pred, self.dataPack.catE)
                    tra_loss = _loss.item()
                    
                    
                    with torch.no_grad():
                        vpred = self._cell(self.dataPack.catVX)
                        val_loss = mse_loss(vpred, self.dataPack.catVE)
                        val_loss =val_loss.item()
                        
                    if '{:.4e}'.format(tra_loss) == 'nan' or '{:.4e}'.format(val_loss) == 'nan':
                        break
                    
                    self.record_arch(tra_loss, val_loss, epoch, tag)
                    
                    _loss.backward()

                    self.optimizer.step()
                    # ho_curr = list(self._cell.ho.parameters())[0].clone()
                    # arch_curr = list(self._cell.arch.parameters())[0].clone()
                    # ho_change = torch.equal(ho_last, ho_curr)
                    # arch_change = torch.equal(arch_last, arch_curr)
                    # print('ho layer remain tag: {}'.format(ho_change))
                    # print('arch layer remain tag: {}'.format(arch_change))
                except:
                    self.logger.exception('{}\nGot an error on optimizing cell.\n{}'.format('!'*50,'!'*50))
                    break
            
            if report:
                self.logger.info('{} Training epoch: {} \t Training loss: {:.4e} \t Validation loss: {:.4e}'.format(tag, epoch, tra_loss, val_loss))
            self.fit_info.loss_list.append(tra_loss)
            self.fit_info.vloss_list.append(val_loss)
            self.epoch_scheduler.step()
        
        
        _ho = cell_ho(self._cell, self.dataPack, force_update=True)
        self.record_ho(_ho)
  
    def load_best(self,):
        self._cell.arch.load_state_dict(self.best_state.arch)
        self._cell.ho.load_state_dict(self.best_state.ho)
    
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
        
        if record_loss <= self.best_loss:
            self.best_loss = record_loss
            self.best_state.arch = copy.deepcopy(self._cell.arch.state_dict())
            self.best_epoch = epoch
            
            if self.best_epoch != 0:
                self.logger.info('@@@@@@ Find new best cell state. @@@@@@')
                self.logger.info('{} Training epoch: {} \t Training loss: {:.4e} \t Validation loss: {:.4e}'.format(tag,epoch, tra_loss, val_loss))
    
    def record_ho(self, _ho):
        self.best_state.ho = copy.deepcopy(_ho.state_dict())
        
    
    
    def benign_ho(self,):
        with torch.no_grad():
            _x = self.dataPack.x[0]
            h = self._cell.hidden_forward(_x)
        map_size = h.size(1)
        H = self.dataPack.catE.size(1)
        ho = fcLayer(map_size,H, self.device)
        ho.active()
        return ho
            
    # def naive_train(self, ):
    #     '''
    #     '''
    #     catX = torch.cat(self.dataPack.x, dim=0)
    #     self.logger.info('*'*50)
    #     self.logger.info('-------Start naive training-------')
    #     for epoch in trange(self.max_epoch):
    #         if next(self._cell.parameters()).is_cuda:
    #             torch.cuda.empty_cache()
    #         try:
    #             if self.batch_training:
    #                 # batch training
    #                 tra_loss = 0    
    #                 train_len = len(self.dataPack.e)
    #                 for _x, _e in zip(self.dataPack.x, self.dataPack.e):
    #                     self.optimizer.zero_grad()
    #                     h = self._cell.hidden_forward(_x)
    #                     _loss, _ = naive_loss(h,_e,self.device)
    #                     tra_loss += _loss.item()
    #                     self._cell.ho = None
    #                     _loss.backward()
    #                     self.optimizer.step()
                        
    #                 tra_loss = tra_loss / train_len
    #                 self.record(tra_loss=tra_loss, epoch=epoch)
    #             else:
    #                 # # whole training
    #                 self.optimizer.zero_grad()
    #                 h = self._cell.hidden_forward(catX)
    #                 # update ho foreach training epoch
    #                 _loss, self._cell.ho = naive_loss(h,self.dataPack.catE,self.device)
    #                 tra_loss = _loss.item()
                    
    #                 self.record(tra_loss=tra_loss, epoch=epoch)
                    
    #                 _loss.backward()
    #                 self.optimizer.step()
                    
    #             self.logger.info('Naive training epoch: {} \t loss: {:.4e}'.format(epoch, tra_loss))
    #             self.loss_list.append(tra_loss)
    #             self.epoch_scheduler.step()
                
    #         except:
    #             self.logger.exception('{}\nGot an error on cellTraining.\n{}'.format('!'*50,'!'*50))
    #             break

   
    # def ua_train(self, ):
    #     catX = torch.cat(self.dataPack.x, dim=0)
    #     for epoch in trange(self.max_epoch):
    #         if self.batch_training:
    #             # batch training
    #             tra_loss = 0    
    #             train_len = len(self.dataPack.e)
    #             for _x, _e in zip(self.dataPack.x, self.dataPack.e):
    #                 self.optimizer.zero_grad()
    #                 h = self._cell.hidden_forward(_x)
    #                 _loss= ua_loss(h,_e,self.device)
    #                 tra_loss += _loss.item()
    #                 self._cell.ho = None
    #                 _loss.backward()
    #                 self.optimizer.step()
                    
    #             tra_loss = tra_loss / train_len
    #             self.record(tra_loss=tra_loss, epoch=epoch)
    #         else:
    #             # # whole training
    #             self.optimizer.zero_grad()
    #             h = self._cell.hidden_forward(catX)
    #             # update ho foreach training epoch
    #             _loss = ua_loss(h,self.dataPack.catE.detach().clone(),self.device)
    #             tra_loss = _loss.item()
    #             self._cell.ho = None
                
    #             self.record(tra_loss=tra_loss, epoch=epoch)
                
    #             _loss.backward()
    #             self.optimizer.step()
                
    #         self.logger.info('Current training epoch: {} \t loss: {:.4e}'.format(epoch, tra_loss))
    #         self.loss_list.append(tra_loss)
    #         self.epoch_scheduler.step()
        
    #     self._cell.load_state_dict(self.best_state)
  
        
    # def ua_train(self, ):
    #     '''
    #     Usually, the mean ua_loss is still so large, need fix!
    #     '''
    #     # attetion! there may be a problem in the implementation of the convergence inequality 
        
    #     catX = torch.cat(self.dataPack.x, dim=0)
    #     print('\n')
    #     # batches = len(self.dataPack.e)
    #     for epoch in trange(self.max_epoch):
        
    #         # tra_loss = 0
    #         # for _x, _e in zip(self.dataPack.x, self.dataPack.e):
                
    #         #     self.optimizer.zero_grad()
    #         #     _, h = self._cell.hidden_forward(_x)
                
    #         #     _loss = ua_loss(h,_e,self.device)
    #         #     _loss.backward()
    #         #     tra_loss += _loss.item()
    #         #     self.optimizer.step()
    #         # tra_loss = tra_loss / batches
  
    #         self.optimizer.zero_grad()
    #         _, h = self._cell.hidden_forward(catX)
    #         _loss = ua_loss(h,self.dataPack.catE,self.device) 
            
    #         tra_loss = _loss.item()
    #         if tra_loss <= self.best_loss:
    #             self.best_loss = tra_loss
    #             self.best_state = self._cell.arch.state_dict()
    #             self.best_epoch = epoch
                            
    #         _loss.backward()
    #         self.optimizer.step()

    #         self.logger.info('Current training epoch: {} \t loss: {:.4e}'.format(epoch, tra_loss))
    #         # self.logger.info('Current training e_resP: {:.6e}'.format(e_resP.item()))
    #         # self.logger.info('Current training e_feaP: {:.6e}'.format(e_feaP.item()))
    #         # self.logger.info('Current training epoch: {} \t Current training loss: {}'.format(epoch, tra_loss))
    #         self.loss_list.append(tra_loss)
    #         self.epoch_scheduler.step()

# def ua_loss(input, error, device = torch.device('cpu'), type = 'matrix'):
#     '''
#     return loss, if the loss is smaller than 0, the convergence condition is satisfied.\n
#     Attention! Result of type 'a' and type 'm' is equal, the computing type 'm' is far faster than type 'a'.\n
#     Attention! For computing this loss, the error is better scalered by normal distribution with mean 0.
#     '''

#     # start_time = time.time()
#     len_fm = input.data.size(1)
#     samples = error.data.size(0)
#     H = error.data.size(1)
#     scale = samples * samples * H

#     # print('-'*50)
#     # print("---init {:.2f} seconds ---".format(time.time() - start_time))
#     # hstart_time = time.time()
    
#     m_innerP = input.T @ input
#     if type == 'algebra':
#         # algebra form, has been checked!   
#         e_featureP = torch.zeros(H).to(device)
#         e_resP = torch.zeros(H).to(device)
#         m_ips = m_innerP.diagonal().view(-1,)
#         e_f = error.T @ input 
#         for h in range(H):
#             e_f_ij = e_f[h, :]
#             e_featureP[h] = (torch.pow(e_f_ij, 2) / m_ips).sum(dim=0) 
#             for i in range(len_fm - 1):
#                 for j in range(i + 1, len_fm):
#                     e_resP[h] += 2 * e_f_ij[i] * e_f_ij[j] * m_innerP[i,j]/ (m_ips[i] * m_ips[j])
#                     # finish computing the res
    
#         e_featureP = e_featureP.sum(dim=0) / scale
#         e_resP = e_resP.sum(dim=0) / scale
#         loss = e_resP -  e_featureP
#     elif type == 'matrix':
#         # matrix form, has been checked!
#         e_f = error.T @ input 
#         loss_H = torch.zeros(H).to(device)
#         m_ips = m_innerP.diagonal().view(1,-1)
#         for h in range(H):
#             e_f_ij = e_f[h,:].reshape(1,-1)
#             ef_prod = e_f_ij / m_ips
#             ef_innerP = ef_prod.T @ ef_prod * m_innerP
#             # e_featureP[h] = ef_innerP.sum() - ef_innerP.diagonal().sum()
#             # e_resP[h] = ef_innerP.diagonal().sum()
#             loss_H[h] = ef_innerP.sum() - 2 * ef_innerP.diagonal().sum()
        
#         loss = loss_H.sum() / scale
#     # we want a greater e_featureP  while a smaller e_resP 
#     return loss


# def naive_loss(input, error, device = torch.device('cpu')):
#     # start_time = time.time()
#     ho = close_ho(device, input,error, report= False)
#     pred = ho(input)
#     loss = mse_loss(pred, error) 
#     # theoretically, mse_loss is equal to torch.pow(torch.linalg.matrix_norm(error - pred), 2) / (samples * H )
#     # new_error = error - pred
#     # loss = torch.linalg.matrix_norm(new_error) / (samples * H )
#     return loss, ho
