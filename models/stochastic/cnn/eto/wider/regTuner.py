# from os import WEXITED
# import torch
# from models.stochastic.cnn.ice.wider.basic import close_ho,io_check,cnn_map,rnn_map
# # from task.TaskLoader import Opt
# from models.stochastic.Base import fcLayer
# import torch
# import gc

from task.TaskLoader import Opt

import torch
import gc
from models.stochastic.cnn.eto.wider.basic import close_ho, cell_ho, cal_eNorm
from task.util import set_logger

import os
import ray
from ray import tune
# from ray.tune import Analysis
from ray.tune.search.ax import AxSearch
from ray.tune.search import ConcurrencyLimiter
import logging
from ray.tune.search.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis
from ray.tune.search.basic_variant import BasicVariantGenerator

from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
from ray.air import FailureConfig

import pickle   

class ReadoutTuner(Opt):
    def __init__(self,):
        super().__init__()
        
        # self.best_loss = float('inf')
        self.best_lambda_reg = 0
    # def config_to_ho(self, hOpt, ho_config, dataPack):
        
        
    def tuning(self, aHyper, _cell, dataPack, wNum):
        '''input: hyper-parameter, dataPack, width_num, device\n
        return: new selected arch_cell, updated dataPack'''


        self.WidthNum = wNum
        self.algo_name = aHyper.ho.tuner.algo_name
        
        self.tuner_dir = os.path.join(dataPack.model_fit_dir, 'tuner', 'WideNet{}'.format(self.WidthNum),'Readout')
        self.analysis_dir = os.path.join(self.tuner_dir, self.algo_name)
        
        rt_config_path = os.path.join(self.analysis_dir, 'rT.config.pkl')
        
        if os.path.exists(self.analysis_dir):
                # analysis = Analysis(self.analysis_dir)
                # best_config = analysis.get_best_config(metric='e_norm', mode='min')
                try:
                    if os.path.exists(rt_config_path):
                        with open(rt_config_path, 'rb') as rt_pkl:
                            rt_loading = pickle.load(rt_pkl)
                            best_config = rt_loading.best_config
                    else:
                        analysis = ExperimentAnalysis(self.analysis_dir)
                        best_config = analysis.get_best_config(metric='e_norm', mode='min')
                except:
                    aHyper.ho.H = dataPack.H
                    arch_tuner = hoTuner(aHyper.ho) 
                    arch_tuner.conduct(dataPack, wNum, _cell)
                    best_config = arch_tuner.best_config
        else:
            # atm = getattr(current_module, '{}Tuner'.format(arch_name))
            aHyper.ho.H = dataPack.H
            arch_tuner = hoTuner(aHyper.ho) 
            arch_tuner.conduct(dataPack, wNum, _cell)
            
            best_config = arch_tuner.best_config

        if not os.path.exists(rt_config_path):
            with open(rt_config_path, 'wb') as rt_pkl:
                rt_config = Opt()
                rt_config.best_config = best_config
                pickle.dump(rt_config, rt_pkl)
            
        self.best_lambda_reg = best_config['lambda_reg']
        
        self.host_dir  = os.path.join(dataPack.model_fit_dir, 'rTuner', 'WideNet{}'.format(self.WidthNum))
        self.logger = set_logger(os.path.join(self.host_dir, 'rTuning.log'),'cellTrainer',level=20)
        self.logger.info('>'*20)
        self.logger.info('Current Net Width: {} Best lambda_reg: {}'.format(self.WidthNum, self.best_lambda_reg))
        self.logger.info('Bef.-tuning ho inverse: {} \t lambda_reg: {}'.format(_cell.ho.inverse, _cell.ho.lambda_reg))
        
        _cell.ho = cell_ho(_cell, dataPack, force_update=True, lambda_reg= self.best_lambda_reg)
        _, rT_eNorm = cal_eNorm(_cell, dataPack)

        self.logger.info('Aft.-tuning ho inverse: {} \t lambda_reg: {}'.format(_cell.ho.inverse, _cell.ho.lambda_reg))
        
        self.logger.info('*'*20)
        self.logger.info('Current Error 2-norm (before preTuning): {}'.format(_cell.last_eNorm))
        self.logger.info('Current Error 2-norm (after preTuning): {}'.format(_cell.pT_eNorm))
        if hasattr(_cell, 'cT_eNorm'):
            self.logger.info('Current Error 2-norm (after cellTraining): {}'.format(_cell.cT_eNorm))
        
        self.logger.info('>'*20)
        self.logger.info('Current Error 2-norm (after hoTuning): {}'.format(rT_eNorm))
        # ho = self.ho_get(_cell,dataPack,_cell.fc_io,ho_config.best_config['lambda_reg'])

        # 计算训练集误差

        _cell.rT_eNorm = rT_eNorm
        _cell.lambda_reg = best_config['lambda_reg']
        
        return _cell


class hoTuner(Opt):
    def __init__(self, arch_opts):
        super().__init__()

        self.merge(arch_opts)

        # self.tuner.metirc = 'e_norm'
        self.metric = 'e_norm'
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 16
        else:
            self.tuner.iters = self.tuner.iters

        self.cpus = 36
        self.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
        }
        self.algo_name = self.tuner.algo_name

    def funcData_gen(self, dataPack, _cell):

        # processing val_loader
        with torch.no_grad():                
            cat_vH = _cell.hidden_forward(dataPack.catVX)
            
            cat_H = []
            for _x in dataPack.x:
                _f = _cell.hidden_forward(_x)
                cat_H.append(_f)
            
            cat_H = torch.cat(cat_H, dim=0)
        
        func_data = Opt()
        func_data.cat_E = dataPack.catE.detach().clone()
        func_data.cat_H = cat_H
        func_data.cat_vE = dataPack.catVE.detach().clone()
        func_data.cat_vH = cat_vH
        func_data.map_size = cat_H.size(1)
        func_data.H = self.H
        func_data.device = self.device
        return func_data

    

    def conduct(self, dataPack, wNum, _cell):
        self.tuner_dir = os.path.join(dataPack.model_fit_dir, 'tuner', 'WideNet{}'.format(wNum),'Readout')
        if not os.path.exists(self.tuner_dir):
            os.makedirs(self.tuner_dir)
            
        #
        self.H = dataPack.H
        self.device = dataPack.device
        func_data = self.funcData_gen(dataPack, _cell)
            
        ray.init(num_cpus=self.cpus, _redis_max_memory=10**9)
        # self.tuner.iters = 80
        if self.algo_name == 'bayes':
            algo = self.search_ax()
        elif self.algo_name == 'tpe':
            algo = self.search_tpe()
        else:
            algo = self.search_randn()

        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(ReadoutCell, data=func_data), resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=tune.TuneConfig(
                search_alg=algo,
                metric=self.metric,
                mode="min",
                num_samples=self.tuner.iters ),
            run_config=RunConfig(
                name=self.algo_name,
                local_dir=self.tuner_dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.iters // 2),
                stop={'training_iteration':self.tuner.iters}
                )
            )
        results = tuner.fit()
        
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner_dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        best_result = results.get_best_result(self.metric, 'min')
        self.best_result = best_result.metrics
        self.best_config  = best_result.config
        
        # self.best_config = analysis.get_best_config()
        # self.best_result = analysis.best_result

    def search_ax(self,):
        self.tuner.name = 'bayes'
        # self.tuner.algo = 'algo'
        ax_search = ConcurrencyLimiter(
            AxSearch(
                metric=self.metric, 
                mode='min', 
                verbose_logging=False
                ), 
            max_concurrent=4)
        # mute the warning and info in the belowing loggers.
        for logger_name in ['ax', 'ax.core.parameter', 'ax.core.parameter', 'ax.service.utils.instantiation', 'ax.modelbridge.dispatch_utils']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        return ax_search

    def search_tpe(self,):
        self.tuner.name = 'tpe'
        # self.tuner.algo = 'algo'
        tpe_search = ConcurrencyLimiter(
            OptunaSearch(
                metric=self.metric, mode='min'
                ), 
            max_concurrent=6
            )
        return tpe_search

    def search_randn(self,):
        self.tuner.name = 'rand'
        rad_search = BasicVariantGenerator(max_concurrent=6)
        return rad_search
        

class ReadoutCell(tune.Trainable):
    # 输入值等，计算得到最终的fitness
    def setup(self, config, data = None):

        self.cat_E = data.cat_E
        self.cat_H = data.cat_H
        self.cat_vE = data.cat_vE
        self.cat_vH = data.cat_vH
        self.H = data.H
        self.map_size = data.map_size
        self.device = data.device
        
        self.lambda_reg = config['lambda_reg']

        self.ho = close_ho(self.device, self.cat_H, self.cat_E,self.lambda_reg)
    
    def step(self,):
        
        cat_p = self.ho(self.cat_vH)
        # loss = self.loss_fn(cat_p, self.d_cate).item()

        cat_e = self.cat_vE - cat_p
        e_norm = torch.linalg.matrix_norm(cat_e).item()

        return {
            # "loss": loss,
            "e_norm": e_norm
        }

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.ho.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.ho.load_state_dict(torch.load(checkpoint_path))
        
