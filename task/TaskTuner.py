'''
Attention !
************
For stochastic model, e.g. ESN, DESN,..., etc. 
They are trained only once that are solving their output-weight in a close form manner. Thus the schedulers based Tuner cannot be implemented into tuning these models, that the tuner will sample the hypers from the config (tuning.dict) only once, and will not tuning sequentially further, causing the parameters 'tuner.iters' is meaningless. 
'''

import os
import sys
from numpy import not_equal

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# import json
import ray
from task.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
# from ray.air.checkpoint import Checkpoint

from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator

import nevergrad as ng

import importlib
# import torch

# import pandas as pd

# import logging


class taskTuner(Opt):
    def __init__(self, opts = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.points_to_evaluate = []
        if 'points_to_evaluate' in self.tuner.dict:
            self.points_to_evaluate = self.tuner.points_to_evaluate
            assert len(self.points_to_evaluate) > 0

        self.best_config = Opt()

    def search_ax(self,):
        self.tuner.name = 'Bayes_Search'
        # self.tuner.algo = 'algo'
    
        ax_search = ConcurrencyLimiter(AxSearch(metric=self.metric, mode='min',verbose_logging = False), max_concurrent=6)

        return ax_search
    
    def search_pso(self,):
        """
        https://github.com/facebookresearch/nevergrad.
        """
        self.tuner.name = 'PSO_Search'
        # self.tuner.algo = 'pso'
        
        _popsize= min((20, self.tuner.iters // 10))
        
        ng_search = NevergradSearch(
            optimizer=ng.optimizers.ConfiguredPSO(
                transform='arctan',
                popsize= _popsize
                ),
            metric=self.metric,
            mode="min",
            points_to_evaluate=self.points_to_evaluate
            )
        return ng_search
    

    def search_tpe(self,):
        '''Tree-structured Parzen Estimator
        https://docs.ray.io/en/master/tune/examples/optuna_example.html
        '''
        self.tuner.name = 'TPE_Search'
        # self.tuner.algo = 'tpe'
    
        tpe_search = ConcurrencyLimiter(
            OptunaSearch(
                metric=self.metric, mode='min',points_to_evaluate=self.points_to_evaluate
                ), 
            max_concurrent=6
            )
        # mute the warning and info in the belowing loggers.
        # for logger_name in ['ax.core.parameter', 'ax.core.parameter','ax.service.utils.instantiation','ax.modelbridge.dispatch_utils']:
        #     logging.getLogger(logger_name).setLevel(logging.ERROR)
        return tpe_search
    
    def search_randn(self,):
        self.tuner.name = 'Rand_Search'
        # self.tuner.algo = 'rand'
    
        rad_search = BasicVariantGenerator(max_concurrent=6)
        # mute the warning and info in the belowing loggers.
        # for logger_name in ['ax.core.parameter', 'ax.core.parameter','ax.service.utils.instantiation','ax.modelbridge.dispatch_utils']:
        #     logging.getLogger(logger_name).setLevel(logging.ERROR)
        return rad_search
    
    # def search_bohb(self,):
    #     """
    #     More efficiency with early stop via Bayesian Optimization HyperBand .
    #     """
    #     self.config_trans()
    #     algo = TuneBOHB(max_concurrent=4)
    #     sched = HyperBandForBOHB()
    #     analysis = tune.run(
    #         self.fitness,
    #         metric="best_loss",
    #         mode="min",
    #         name='BOHB_Search',
    #         search_alg=algo,
    #         scheduler=sched,
    #         config=self.tuning,
    #         resources_per_trial=self.resource,
    #         num_samples=self.tuner.iters,
    #         # local_dir=self.opts.task_dir,
    #         verbose=1
    #     )
    #     return analysis


class StocHyperTuner(taskTuner):
    '''
    https://docs.ray.io/en/master/tune/examples/tune-pytorch-cifar.html
    '''
    def __init__(self, opts, logger, subPack):
        # self.xxxx = opts.xxxx
        # super().__init__(self.opts.tunings,search_times)
        super().__init__(opts)

        self.logger = logger
        
        self.hyper.H = subPack.H
        self.hyper.sid = subPack.index
        self.hyper.sid_name = subPack.name
        self.hyper.cid = 'T'
        
        self.train_data = subPack.train_data
        self.valid_data = subPack.valid_data
        self.batch_size = subPack.batch_size
        # for testing
        # self.tuner.metric = 'vrmse' # to do: comment
        self.metric = 'vrmse'
        
        if 'algo' not in self.tuner.dict:
            self.algo_name = 'rand'
        elif self.tuner.algo not in ['tpe','pso', 'rand']:
            raise ValueError('Non supported tuning algo: {}'.format(self.tuner.algo))
        else:
            self.algo_name = self.tuner.algo
        
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 20
            
        if 'resource' not in self.tuner.dict:
            self.resource = {
            "cpu": 10, 
            "gpu": 1  # set this for GPUs
        }
        else:
            self.resource = self.tuner.resource
            
        self.loss_upper_bound = 99999
        
        if self.algo_name == 'tpe':
            self.algo_func = self.search_tpe()
        elif self.algo_name == 'pso':
            self.algo_func = self.search_pso()
        elif self.algo_name == 'rand':
            self.algo_func = self.search_randn()
            
        
            

    def fitness(self, config):
        
        # loaded_checkpoint = session.get_checkpoint()

        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)
        model = model(_hyper, self.logger)
        # if loaded_checkpoint:
        #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
        #         model_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        #     model.load_state_dict(model_state)

        fit_info = model.xfit(self.train_data, self.valid_data,)
        trmse, vrmse = fit_info.trmse, fit_info.vrmse

        trmse = trmse if trmse < self.loss_upper_bound else self.loss_upper_bound
        vrmse = vrmse if vrmse < self.loss_upper_bound else self.loss_upper_bound

        # checkpoint_dir = os.makedirs(os.path.join(self.tuner.dir, 'checkpoint'))
        # torch.save(model)

        session.report({
            'trmse': trmse,
            'vrmse': vrmse,
        })

    def conduct(self,):
        ray.init(num_cpus=30)
        # self.tuner.iters = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.fitness), 
                resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=
            tune.TuneConfig(
            # name=self.algo_name,
            search_alg=self.algo_func,
            # resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.iters,
            # local_dir=self.tuner.dir,
            # verbose=1,
            # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                local_dir=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.iters // 2)
                # log_to_file=True
            )
        )
        
        results = tuner.fit() 
            
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'min')
        self.best_config.merge(best_result.config)
        self.best_result = best_result.metrics
        self.logger.info("Best config is:", self.best_config.dict)
        
        return self.best_config
