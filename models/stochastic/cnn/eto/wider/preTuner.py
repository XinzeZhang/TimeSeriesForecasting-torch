from time import sleep
from matplotlib.pyplot import cla
from numpy import fabs
from ray.cloudpickle.cloudpickle import cell_set
from ray.tune.trainable import Trainable
from task.TaskLoader import Opt

from models.stochastic.cnn.eto.wider.basic import close_ho, io_check, Arch_dict

import ray
from ray import tune,air
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search import ConcurrencyLimiter

from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
from ray.air import FailureConfig
import logging

import torch
import os
import torch.nn as nn
import numpy as np

import copy
import gc

import sys
# from ray.tune import Analysis
from ray.tune import ExperimentAnalysis
current_module = sys.modules[__name__]

import pickle


class PreTuner(Opt):
    def __init__(self, arch_choice):
        super().__init__()
        
        self.arch_choice = arch_choice
        # self.algo_name = 'tpe'
        # self.arch_choice = ['esn']
        # self.arch_choice = ['esn']
        
    def tuning(self, aHyper, data_pack, WidthNum, device):
        '''input: hyper-parameter, data_pack, width_num, device\n
        return: new selected arch_cell, updated data_pack'''
    
        self.arch_config_path = os.path.join(data_pack.model_fit_dir, 'tuner', 'WideNet{}'.format(
            WidthNum), 'Arch.config.pkl')
        
        if os.path.exists(self.arch_config_path):
            with open(self.arch_config_path, 'rb') as arch_pkl:
                arch_config = pickle.load(arch_pkl)
                aHyper.dict[arch_config.name].hyper.device = device    
        else:
            arch_config = Opt()
            arch_config.e_norm = float('inf')
            
            for arch_name in self.arch_choice:
                aHyper.dict[arch_name].hyper.device = device         
                algo_name = aHyper.dict[arch_name].tuner.algo_name
                # aHyper.dict[arch_name].tuner.algo_name = self.algo_name
                
                tuner_dir = os.path.join(data_pack.model_fit_dir, 'tuner', 'WideNet{}'.format(
                WidthNum), 'Arch.{}'.format(arch_name))
                aHyper.dict[arch_name].tuner.curr_dir = tuner_dir
                analysis_dir = os.path.join(tuner_dir,'{}'.format(algo_name))
                subarch_path = os.path.join(analysis_dir, 'subarch.config.pkl')

                if os.path.exists(analysis_dir):
                    try:
                        if os.path.exists(subarch_path):
                            with open(subarch_path, 'rb') as subarch_pkl:
                                subarch_loading = pickle.load(subarch_pkl)
                                best_config = subarch_loading.best_config
                                arch_loss = subarch_loading.arch_loss
                        else:
                            analysis = ExperimentAnalysis(analysis_dir)
                            best_config = analysis.get_best_config(metric='e_norm', mode='min')
                            
                            df = analysis.dataframe(metric='e_norm', mode='min')
                            arch_loss = df['e_norm'].min()
                    except:
                        arch_tuner = ArchTuner(aHyper.dict[arch_name]) 
                        arch_tuner.conduct(data_pack)
                        arch_loss = arch_tuner.best_result['e_norm']
                        best_config = arch_tuner.best_config
                else:
                    # atm = getattr(current_module, '{}Tuner'.format(arch_name))
                    arch_tuner = ArchTuner(aHyper.dict[arch_name]) 
                    arch_tuner.conduct(data_pack)
                    arch_loss = arch_tuner.best_result['e_norm']
                    best_config = arch_tuner.best_config
                
                if not os.path.exists(subarch_path):             
                    with open(subarch_path, 'wb') as subarch_pkl:
                        subarch_config = Opt()
                        subarch_config.name = arch_name
                        subarch_config.best_config = best_config
                        subarch_config.arch_loss = arch_loss                    
                        pickle.dump(subarch_config, subarch_pkl)
                
                if arch_loss <= arch_config.e_norm:
                    arch_config.best_config = best_config
                    arch_config.name = arch_name
                    arch_config.e_norm = arch_loss

            # saving the arch_cofig results to its path.
            with open(self.arch_config_path, 'wb') as arch_pkl:
                pickle.dump(arch_config, arch_pkl)

        arch_config, best_arch, fc_io = self.config_to_arch(aHyper.dict[arch_config.name], arch_config, data_pack)

        return arch_config, best_arch, fc_io

    def config_to_arch(self, cOpt, arch_config, data_pack):
        hyper = Opt(init=cOpt.hyper)

        hyper.input_dim = data_pack.input_dim
        
        fc_io = arch_config.best_config['fc_io']
        
        arch_config.best_config.pop('fc_io')
        hyper.update(arch_config.best_config)
        
        arch_config.hyper = hyper
        
        arch_func = Arch_dict[arch_config.name]
        best_arch = arch_func(**hyper.dict).to(hyper.device)
        best_arch.freeze()
        return arch_config, best_arch, fc_io


class ArchTuner(Opt):
    def __init__(self, arch_opts):
        super().__init__()

        self.merge(arch_opts)
    
        self.metric = 'e_norm'
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 16

        if 'resource' not in self.tuner.dict:
            self.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
        }
        else:
            self.resource = self.tuner.resource

        self.cpus = 36
                
        self.arch_name = arch_opts.name
        self.algo_name = self.tuner.algo_name
        # self.arch_func = Arch_dict[self.arch_name]
        self.loss_fn = nn.MSELoss()
        self.tuner_dir = self.tuner.curr_dir
        

    def space_sampling(self,):
        config = Opt(init=self.tuning)
        for key in self.tuning.dict:
            config.dict[key] = self.tuning.dict[key].sample()
        return config.dict
    
    def funcData_gen(self, data_pack):
        func_data = Opt()
        with torch.no_grad():
            func_data.d_x = [_i.detach().clone() for _i in data_pack.x] # ensure the arch are selected based on input x
            func_data.d_cate = data_pack.catE.detach().clone()
            func_data.d_catx = data_pack.catX.detach().clone()
            func_data.d_catvx = data_pack.catVX.detach().clone()
            func_data.d_catve = data_pack.catVE.detach().clone()
            
        func_data.merge(self, ['hyper', 'arch_name',
                        'device'])
        func_data.merge(data_pack, ['steps', 'input_dim', 'H', 'xm_size'])
        
        return func_data
    
    def once_sample(self, data_pack):
        # self.tuner_dir = os.path.join(data_pack.model_fit_dir, 'tuner', 'WideNet{}'.format(
        #     WidthNum), 'Arch.{}'.format(self.arch_name))        
        self.best_config = self.space_sampling()
            
        func_data = self.funcData_gen(data_pack)
        
        tuning = Cell_trial(self.best_config, func_data)
        result = tuning.step()
        self.best_result = result
        

    def _conduct(self, data_pack):
        if self.tuner.algo_name == 'tpe':
            algo = self.search_tpe()
        elif self.tuner.algo_name == 'bayes':
            algo = self.search_ax()
        else:
            algo = self.search_randn()

        func_data = self.funcData_gen(data_pack)
            
        ray.init(num_cpus=self.cpus, _redis_max_memory=10**9)
        # self.tuner.iters = 80
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(tuningCell, data=func_data), resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=tune.TuneConfig(
                search_alg=algo,
                metric=self.metric,
                mode="min",
                num_samples=self.tuner.iters),
            run_config=RunConfig(
                name=self.tuner.algo_name,
                local_dir=self.tuner.curr_dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.iters // 2),
                stop={'training_iteration':self.tuner.iters},
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=0,
                    checkpoint_at_end = False
                ),
                sync_config=tune.SyncConfig(
                    syncer=None
                )
                ),
            
            )
        results = tuner.fit()
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.curr_dir, '{}.trial.csv'.format(self.tuner.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'min')
        self.best_result = best_result.metrics
        self.best_config  = best_result.config

        # analysis = tune.run(
        #     tune.with_parameters(tuningCell, data=func_data),
        #     name=#self.tuner.name,
        #     search_alg=algo,
        #     config=self.tuning.dict,
        #     resources_per_trial=self.resource,
        #     metric=self.metric,
        #     mode="min",
        #     num_samples=self.tuner.iters,
        #     local_dir=self.tuner_dir,
        #     verbose=1,
        #     stop={
        #         "training_iteration": self.tuner.iters
        #     },
        #     raise_on_failed_trial = False
        # )

        # self.best_config = analysis.get_best_config()
        # self.best_result = analysis.best_result
        # ray.shutdown()

        
        
    def search_ax(self,):
        #self.tuner.name = 'bayes'
        #self.tuner.algo = 'algo'
        ax_search = ConcurrencyLimiter(
            AxSearch(
                metric=self.metric, 
                mode='min', 
                verbose_logging=False
                ), 
            max_concurrent=4)
        # mute the warning and info in the belowing loggers.
        for logger_name in ['ax', 'ax.core.parameter', 'ax.service.utils.instantiation', 'ax.modelbridge.dispatch_utils']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        return ax_search

    def search_tpe(self,):
        #self.tuner.name = 'tpe'
        #self.tuner.algo = 'algo'
        tpe_search = ConcurrencyLimiter(
            OptunaSearch(
                metric=self.metric, mode='min'
                ), 
            max_concurrent=6
            )
        return tpe_search

    def search_randn(self,):
        #self.tuner.name = 'rand'
        rad_search = BasicVariantGenerator(max_concurrent=6)
        return rad_search
    
    def conduct(self, data_pack):
        self.device = data_pack.device
        if not os.path.exists(self.tuner_dir):
            os.makedirs(self.tuner_dir)
        if self.tuner.iters  == 1:
            self.once_sample(data_pack)
        else:
            self._conduct(data_pack)



class tuningCell(tune.Trainable):
    def setup(self, config, data=None):

        self.d_x = data.d_x
        self.d_cate = data.d_cate
        self.d_catx = data.d_catx
        
        self.d_catvx = data.d_catvx 
        self.d_catve = data.d_catve

        self.device = data.device

        self.hyper = data.hyper  # check xm_size if fc_io; check steps.
        self.arch_name = data.arch_name
        self.steps = data.steps

        
        self.H = data.H

        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.fc_io = None
        _hyper.update(config)
        _hyper.input_dim = data.input_dim # must check this for arch mixed-up
        
        self.fc_io = _hyper.fc_io # put the fc_io into the tuning
 
        self.xm_size = data.xm_size

        if 'fc_io' in _hyper.dict:
            _hyper.dict.pop('fc_io')
        self.model = Arch_dict[self.arch_name](**_hyper.dict).to(self.device)
        self.model.freeze()
        # self.ho = self.gen_ho(_hyper)
        # self.ho.freeze()

        self.loss_fn = nn.MSELoss()

    # def gen_ho(self, _hyper):
    #     if self.arch_name == 'cnn':
    #         map_size = cnn_map(_hyper, self.steps)
    #     elif self.arch_name == 'esn':
    #         map_size = rnn_map(_hyper)
    #     else:
    #         raise ValueError('Non-supported arch type: {}'.format(self.arch_name)
    #                          )

    #     if self.fc_io:
    #         map_size += self.xm_size
    #     ho = fcLayer(map_size, self.H, self.device)
    #     ho.freeze()
    #     return ho


    def step(self,):
        cat_h = []
        for _x in self.d_x:
            _, h = self.model(_x)
            cat_h.append(h)
        cat_h = torch.cat(cat_h, dim=0)
        cat_h = io_check(self.fc_io, cat_h, self.d_catx)

        # weight, bias = close_ho(self.device, cat_h, self.d_cate)
        # self.ho = fcLayer(cat_h.size(1), self.d_cate.size(1), self.device)
        # self.ho.update(weight, bias)
        # self.ho.freeze()
        try:
            self.ho = close_ho(self.device, cat_h, self.d_cate)
        except:
            self.ho = close_ho(self.device, cat_h, self.d_cate, lambda_reg=10)
        # cat_p = self.ho(cat_h)
        # loss = self.loss_fn(cat_p, self.d_cate).item()
        
        _, cat_vh = self.model(self.d_catvx)
        cat_vh = io_check(self.fc_io, cat_vh, self.d_catvx)
        cat_vp = self.ho(cat_vh)
        
        
        # cat_e = self.d_cate - cat_p
        cat_e = self.d_catve - cat_vp
        e_norm = torch.linalg.matrix_norm(cat_e).item()

        return {
            # "loss": loss,
            "e_norm": e_norm
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


class Cell_trial(tuningCell):
    def __init__(self, config = None, data = None):
        self.setup(config,data)
        
    
