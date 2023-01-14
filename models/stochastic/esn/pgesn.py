from models.stochastic.esn.GrowingESN import Incremental_ESN
import copy
from models.stochastic.Base import esnLayer
import torch
import torch.nn as nn
import numpy as np
from task.TaskLoader import Opt
from tqdm import trange, tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# from re import S
# from numpy.lib.function_base import select
# from sko.PSO import PSO

# from models.stochastic.esn.DeepESN import Deep_ESN
# from models.stochastic.esn.tuner import PSOTuner


class PSO_Gesn(Incremental_ESN):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts=opts, logger=logger)

    def Layer_select(self, train_loader, Error):
        # bestLayer = esnLayer(
        #     init='svd',
        #     hidden_size=self.Hidden_Size,
        #     input_dim=self.Input_dim,
        #     nonlinearity=self.opts.nonlinearity,
        #     leaky_r=self.opts.leaky_r,
        #     weight_scale=self.opts.weight_scaling,
        #     iw_bound=self.opts.iw_bound,
        #     hw_bound=self.opts.hw_bound,
        #     device=self.device
        # )
        # bestLayer.freeze()

        # pso_para = self.opts.pso

        tuner = PSOTuner(self.Hidden_Size, self.Input_dim,
                         self.device, train_loader, Error, self.opts)
        bestLayer = tuner.conduct()
        return bestLayer

    def init_arch(self, ):
        self.res_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])

        for w in range(self.Subreservoir_Size):
            readout_size = self.Hidden_Size
            if self.fc_io:
                readout_size += self.Input_dim * \
                    (self.Time_steps - self.Discard_steps)
            readout = nn.Linear(readout_size, self.Output_dim)
            self.ho_list.append(readout)
            ## Attention!!! There is another manner of implementation, which
            # fc_io is only calculated once and independent to all sub-reservoirs.

    def xfit(self, train_loader, val_loader):
        min_vrmse = float('inf')
        x = []
        y = []
        for batch_x, batch_y in tqdm(train_loader):
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
        vpred = torch.zeros_like(val_y).to(self.device)

        for i in trange(self.Subreservoir_Size):
            subs = i + 1
            self.logger.info('-'*55)
            self.logger.info(
                'Subs size: {} \t Reservoir size: {}'.format(
                    subs, self.Hidden_Size * subs))

            layer_esn = self.Layer_select(
                train_loader, Error)     # pso 得到优化过后的 layer_esn

            self.res_list.append(layer_esn)
            h_state, x, y = self.batch_transform(train_loader, i)
            self.update_readout(h_state, x, Error, i)
            pred = pred + self.ho_list[i](self.io_check(h_state, x))
            #增量更新方式
            #训练
            Error = y - pred
            #存储已经得到的预测值
            loss = np.sqrt(self.loss_fn(pred, y).item())
            self.fit_info.loss_list.append(loss)

            vh_state, val_x, val_y = self.batch_transform(val_loader, i)
            vpred = vpred + self.ho_list[i](self.io_check(vh_state, val_x))

            vloss = np.sqrt(self.loss_fn(vpred, val_y).item())
            self.fit_info.vloss_list.append(vloss)

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

    # def xfit(self, train_loader, val_loader):
    #     min_vrmse = float('inf')

    #     x = []
    #     y = []
    #     for batch_x, batch_y in tqdm(train_loader):
    #         batch_y = batch_y.to(self.device)
    #         x.append(batch_x)
    #         y.append(batch_y)
    #     x = torch.cat(x, dim=0)
    #     y = torch.cat(y, dim=0)
    #     Error = y.detach().clone()
    #     pred = torch.zeros_like(y).to(self.device)

    #     val_y = []
    #     for _, batch_y in tqdm(val_loader):
    #         batch_y = batch_y.to(self.device)
    #         val_y.append(batch_y)
    #     val_y = torch.cat(val_y, dim=0)
    #     vpred = torch.zeros_like(val_y).to(self.device)

    #     for i in trange(self.Subreservoir_Size):
    #         subs = i + 1
    #         self.logger.info('-'*55)
    #         self.logger.info(
    #             'Subs size: {} \t Reservoir size: {}'.format(
    #                 subs, self.Hidden_Size * subs))

    #         # 对第i个进行优化

    #         layer_esn_best = self.Layer_select(x, Error)
    #         self.res_list[i] = layer_esn_best
    #         h_state = layer_esn_best.forward(x)

    #         self.update_readout(h_state, x, Error, i)
    #         pred = pred + self.ho_list[i](self.io_check(h_state, x))
    #         #增量更新方式
    #         #训练
    #         Error = y - pred
    #         #存储已经得到的预测值
    #         loss = np.sqrt(self.loss_fn(pred, y).item())
    #         self.fit_info.loss_list.append(loss)

    #         vh_state, val_x, val_y = self.batch_transform(val_loader, i)
    #         vpred = vpred + self.ho_list[i](self.io_check(vh_state, val_x))

    #         vloss = np.sqrt(self.loss_fn(vpred, val_y).item())
    #         self.fit_info.vloss_list.append(vloss)

    #         if vloss < min_vrmse:
    #             min_vrmse = vloss
    #             self.best_res_idx = i
    #             self.logger.info('****** Found new best state ******')
    #             # self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))
    #         self.logger.info('Best VRMSE: {:.8f} \t Best Res_idx: {} \t  Training RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
    #             min_vrmse, self.best_res_idx, loss, vloss))

    #     self.fit_info.trmse = self.fit_info.loss_list[self.best_res_idx]
    #     self.fit_info.vrmse = min_vrmse

    #     return self.fit_info


class PSOTuner(Opt):
    def __init__(self, Hidden_Size, Input_dim, device, data_loader, Error, opts):
        # 初始化参数
        # diag_opts : lb,ub,dim;fc_io;
        # pso_para : w,c1,c2;pop,max_iter;
        super().__init__()

        # 参数初始化
        self.data_loader = data_loader
        self.Error = Error
        self.pso_para = opts.pso
        # self.pso_para = pso_para

        self.opts = opts
        self.Hidden_Size = Hidden_Size
        self.Input_dim = Input_dim
        self.device = device

        # ih,svd_u,svd_v初始化
        self.ih = self.ih_init()
        self.svd_u, self.svd_v = self.svd_init()

        self.loss_fn = nn.MSELoss()

        # personal best location of every particle in history
        self.pbest_individual = [None] * self.pso_para.pop
        self.pbest_fit = [9999] * self.pso_para.pop
        # global best location for all particles
        self.gbest_individual = individual(
            self.opts, self.Hidden_Size, self.Input_dim, self.device)
        self.gbest_fit = 99999  # global best location for all particles

        if isinstance(self.opts.hw_bound, tuple):
            v_high = self.opts.hw_bound[1] - self.opts.hw_bound[0]
        elif isinstance(self.opts.hw_bound, float):
            v_high = self.opts.hw_bound - (- self.opts.hw_bound[0])
        else:
            raise ValueError('Invalid hw_bound: {}'.format(self.opts.hw_bound))
        # speed of particles
        self.V = np.random.uniform(-v_high, v_high,
                                   (self.Hidden_Size, self.pso_para.pop))

    def ih_init(self):
        ih = nn.Linear(self.Input_dim, self.Hidden_Size,
                       bias=False).to(self.device)
        w_ih = torch.empty(self.Hidden_Size, self.Input_dim).to(self.device)
        if isinstance(self.opts.iw_bound, tuple):
            nn.init.uniform_(
                w_ih, self.opts.hw_bound[0], self.opts.hw_bound[1])
        elif isinstance(self.opts.hw_bound, float):
            nn.init.uniform_(w_ih, -self.opts.hw_bound, self.opts.hw_bound)
        else:
            raise ValueError('Invalid iw_bound: {}'.format(self.opts.iw_bound))
        ih.weight = nn.Parameter(w_ih)

        return ih

    def svd_init(self):
        svd_u = torch.empty(
            self.Hidden_Size, self.Hidden_Size)
        svd_v = torch.empty(
            self.Hidden_Size, self.Hidden_Size)
        # 填充正交矩阵，非零元素依据均值0，标准差std的正态分布生成
        nn.init.orthogonal_(svd_u)
        nn.init.orthogonal_(svd_v)

        return svd_u, svd_v

    def initPop(self):
        self.individuals = []
        for i in range(self.pso_para.pop):
            individual_tem = individual(
                self.opts, self.Hidden_Size, self.Input_dim, self.device)
            individual_tem.cal_fitness(
                self.ih, self.svd_u, self.svd_v, self.data_loader, self.Error, self.loss_fn)
            self.individuals.append(individual_tem)
            self.pbest_individual[i] = individual_tem
        self.update_gbest()

    def update_V(self, i):
        r1 = np.random.rand()
        r2 = np.random.rand()
        self.V[:, i] = self.pso_para.w * self.V[:, i] + \
            self.pso_para.cp * r1 * (self.pbest_individual[i].svd_s - self.individuals[i].svd_s) + \
            self.pso_para.cg * r2 * \
            (self.gbest_individual.svd_s - self.individuals[i].svd_s)

    def update_individual(self, i):
        svd_s_new = self.individuals[i].svd_s + self.V[:, i]

        if isinstance(self.opts.hw_bound, tuple):
            svd_s_new = np.clip(
                svd_s_new, self.opts.hw_bound[0], self.opts.hw_bound[1])
        elif isinstance(self.opts.hw_bound, float):
            svd_s_new = np.clip(
                svd_s_new, -self.opts.hw_bound, self.opts.hw_bound)
        else:
            raise ValueError('Invalid hw_bound: {}'.format(self.opts.hw_bound))

        self.individuals[i].svd_s = svd_s_new
        self.individuals[i].cal_fitness(
            self.ih, self.svd_u, self.svd_v, self.data_loader, self.Error, self.loss_fn)

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        for i in range(self.pso_para.pop):
            if self.individuals[i].fitness < self.pbest_fit[i]:
                # self.pbest_individual[i] = copy.deepcopy(self.individuals[i])
                self.pbest_individual[i].svd_s = self.individuals[i].svd_s
                self.pbest_individual[i].fitness = self.individuals[i].fitness
                self.pbest_fit[i] = self.individuals[i].fitness

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        min_index = np.argmin(np.array(self.pbest_fit))
        if isinstance(min_index, list):
            index = np.argmin(np.array(self.pbest_fit))[0]
        else:
            index = min_index
        # self.gbest_individual = copy.deepcopy(self.individuals[index])
        self.gbest_individual.svd_s = self.individuals[index].svd_s
        self.gbest_individual.fitness = self.individuals[index].fitness
        self.gbest_fit = self.individuals[index].fitness

    def conduct(self):
        self.initPop()
        for iter_num in range(self.pso_para.max_iter):
            for i in range(self.pso_para.pop):
                self.update_V(i)
                self.update_individual(i)
            self.update_pbest()
            self.update_gbest()

        layer_esn_best = esnLayer(
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
        hh = self.gbest_individual.get_hh(self.svd_u, self.svd_v)
        layer_esn_best.esnCell.hh = hh
        layer_esn_best.esnCell.ih = self.ih
        layer_esn_best.freeze()

        return layer_esn_best


class individual():
    def __init__(self, opts, Hidden_Size, Input_dim, device, nonlinearity=''):
        self.opts = opts    #
        self.Hidden_Size = Hidden_Size
        self.Input_dim = Input_dim
        self.device = device
        # self.res_index = res_index
        self.svd_s = self.svd_s_init()     # 没有对角化,numpy
        self.pred_error = None

        self.fitness = 0

        if self.opts.nonlinearity == 'tanh':
            self.act_f = torch.tanh
        elif self.opts.nonlinearity == 'relu':
            self.act_f = torch.relu
        elif self.opts.nonlinearity == 'sigmoid':
            self.act_f = torch.sigmoid
        else:
            raise ValueError(
                "Unknown nonlinearity '{}'".format(nonlinearity))

        self.Time_steps = opts.steps
        self.Output_dim = opts.H
        self.readout_size = self.Hidden_Size
        if self.opts.fc_io:
            self.readout_size += self.Input_dim * \
                (self.Time_steps - self.opts.discard_steps)

    def svd_s_init(self):
        svd_s = torch.empty(self.Hidden_Size)
        if isinstance(self.opts.hw_bound, tuple):
            nn.init.uniform_(
                svd_s, self.opts.hw_bound[0], self.opts.hw_bound[1])
        elif isinstance(self.opts.hw_bound, float):
            nn.init.uniform_(svd_s, -self.opts.hw_bound, self.opts.hw_bound)
        else:
            raise ValueError('Invalid hw_bound: {}'.format(self.opts.hw_bound))
        return svd_s.numpy()

    def cal_fitness(self, ih, svd_u, svd_v, data_loader, Error, loss_fn):
        # calculate loss for every x in X
        hh = self.get_hh(svd_u, svd_v)
        h_state, x = self.batch_transform(data_loader, ih, hh)

        ho = self.update_readout(h_state, x, Error)
        H_State = self.io_check(h_state, x)
        pred = ho(H_State)
        tloss = np.sqrt(loss_fn(pred, Error).item())
        self.fitness = tloss

    def get_hh(self, svd_u, svd_v):
        #生成对角化矩阵
        hh = nn.Linear(self.Hidden_Size, self.Hidden_Size,
                       bias=False).to(self.device)
        svd_s_temp = torch.diag(torch.Tensor(self.svd_s))
        assert len(svd_s_temp.size()) == 2

        Hidden_weight = svd_u.mm(svd_s_temp).mm(svd_v)
        Hidden_weight = self.opts.weight_scaling * Hidden_weight
        Hidden_weight = Hidden_weight.to(self.device)
        hh.weight = nn.Parameter(Hidden_weight)
        return hh

    def batch_transform(self, data_loader, ih, hh):
        h_states = []
        x = []
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(self.device)
            _h_states = self.reservior_transform(batch_x, ih, hh)
            h_states.append(_h_states)
            x.append(batch_x)
        h_states = torch.cat(h_states, dim=0)
        x = torch.cat(x, dim=0)
        return h_states, x

    def reservior_transform(self, x, ih, hh):
        samples, time_steps = x.shape[0], x.shape[2]
        last_state = torch.zeros(samples, self.Hidden_Size).to(self.device)
        for t in range(time_steps):
            current_input = x[:, :, t]
            current_state = ih(current_input) + hh(last_state)
            current_state = self.act_f(current_state)
            last_state = (1 - self.opts.leaky_r) * last_state + \
                self.opts.leaky_r * current_state
        return last_state

    def io_check(self, hidden, x):
        if self.opts.fc_io:
            _x = x[:, :, self.opts.discard_steps:]
            Hidden = torch.cat(
                (torch.flatten(_x, start_dim=1), hidden), dim=1)
            return Hidden
        else:
            return hidden

    def update_readout(self, h_state, x, y):
        readout = nn.Linear(self.readout_size, self.Output_dim).to(self.device)
        _Hidden = self.io_check(h_state, x)
        W, tag = self.solve_output(_Hidden, y)
        readout.bias = nn.Parameter(W[:, 0], requires_grad=False)
        readout.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        return readout

    def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self.device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (self.opts.lambda_reg * torch.eye(HTH.size(0))).to(self.device)
        A = HTH + I
        # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            W = torch.mm(torch.linalg.pinv(A.cpu()),
                         HTY.cpu()).t().to(self.device)
        return W, tag
