import os, sys
from re import search
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import importlib


from task.TaskLoader import Opt
from task.parser import get_parser

class StageTuner(Opt):
    def __init__(self, hyper_opts,search_times):
        # super().__init__(hyper_opts)

        self.best_config = None

        self.model_train = hyper_opts.training     #等同于D:\A_study1\Torch-Forecasting\models\config (deprecated).py文件中的training,有些模型需要，有些不需要

        self.search_configs = hyper_opts.tuning.dict
        # self.search_configs = {
        #     'hidden_size':tune.qrandint(32, 256, 32)
        # }

        self.search_times = search_times
        
        self.fitness_iteration = hyper_opts.search_train_epoch if self.model_train else 1 # default with 50

        self.cores = hyper_opts.cores # default with 2
        self.cards = hyper_opts.cards # default with 0.25
        
        self.opts = Opt()
        self.opts.cv = -1
        print(self.search_configs)
        ray.init(num_cpus=4)

    
    def _fitness(self, config):
        '''Loading the hyper-parameters in config, then return a score'''
        score = 0
        return score
    
    def fitness(self, config):
        '''Loading the hyper-parameters in config, then return a score'''

        if self.model_train:
            best_score = float('inf')
            for i in range(self.fitness_iteration):
                score = self._fitness(config)
                if score <= best_score:
                    best_score = score
                tune.report(best_loss=best_score, current_loss=score)
        else:
            best_score = self._fitness(config)
            tune.report(best_loss=best_score)

    def search_stat(self, analysis):
        self.tune = True
        self.best_config = analysis.get_best_config()
        print("Best config is:", analysis.best_config)
        print('Best loss is: {:.4f}'.format(
            analysis.best_result['best_loss']))

    def config_trans(self,):
        for key in self.search_configs.keys():
            if isinstance(self.search_configs[key], list):
                self.search_configs[key] = tune.choice(
                    self.search_configs[key])

    def search_base(self,):
        analysis = tune.run(
            self.fitness,
            config=self.search_configs
           
            )

        return analysis


    def search_pbt(self,):
        """
        More effective via population based search
        """

        sched = PopulationBasedTraining(
            time_attr='training_iteration',
            perturbation_interval=30,
            hyperparam_mutations=self.search_configs,
            resample_probability=0.5,
        )

        analysis = tune.run(
            self.fitness,
            name='PBT_Search',
            scheduler=sched,
            metric="best_loss",
            mode="min",
            config=self.search_configs,
            resources_per_trial={
                "cpu": self.cores,
                "gpu": self.cards  # set this for GPUs
            },
            stop={
                'training_iteration': self.search_times
            },
            # num_samples=self.search_configs.population,
            num_samples = 5,
            # local_dir=self.opts.task_dir,
            verbose=1
        )

        self.search_stat(analysis)
        return analysis

    def search_bohb(self,):
        """
        More efficiency with early stop via Bayesian Optimization HyperBand .
        """

        self.config_trans()

        algo = TuneBOHB(max_concurrent=4)
        sched = HyperBandForBOHB()
        analysis = tune.run(
            self.fitness,
            metric="best_loss",
            mode="min",
            name='BOHB_Search',
            search_alg=algo,
            scheduler=sched,
            resources_per_trial={
                "cpu": self.cores,
                "gpu": self.cards  # set this for GPUs
            },
            num_samples=self.search_times,
            config=self.search_configs,
            # local_dir=self.opts.task_dir,
            verbose=1
        )

        self.search_stat(analysis)
        return analysis
 
class ReadoutTuner(StageTuner):
    def __init__(self, hyper_opts, tra, val, device):
        "tra:train_state、object"
        "val:val_state、object"
        super().__init__(hyper_opts)

        self.device = device
        self.state = tra.state
        self.E = tra.E
        
        self.v_state = val.state
        self.v_E = val.E 
        
        self.tra_size = self.state.shape[0]
        self.val_size = self.v_state.shape[0]
        self.total_size = self.tra_size + self.val_size
    
    def solve_output(self, Hidden_States, y, lambda_reg=1.0):
        '''
        sovle the output with ridge regression, lambda_reg (default) = 1.0
        '''
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self.device), Hidden_States), dim=1)


        col = t_hs.size(1)
        HTH = torch.mm(t_hs.t(), t_hs)
        HTY = torch.mm(t_hs.t(), y)

        I = (lambda_reg * torch.eye(HTH.size(0))).to(self.device)
        A = HTH + I

        orig_rank = torch.linalg.matrix_rank(A.float()).item()

        tag = 'Inverse' if orig_rank == col else 'Pseudo-inverse'

        assert tag == 'Inverse'
        
        if tag == 'Inverse':
            W = torch.linalg.solve(A, HTY)
        else:
            W = torch.mm(torch.linalg.pinv(A.cpu()),HTY.cpu()).to(self.device)
            
        # self.logger.info('Solving Method: {} \t L2 regular: {}'.format(tag, 'True' if self.lambda_reg != 0 else 'False'))
        return W, tag
    
    def _fitness(self, config):
        '''To do'''
        lambda_reg = config['lambda_reg']
        W, _ = self.solve_output(self.state, self.E, lambda_reg)
        loss = torch.dist(self.v_state @ W, self.v_E).item()
        # another loss:
        # loss = self.val_size / self.total_size * torch.dist((self.v_state @ W, self.v_E).item() + self.tra_size / self.total_size * torch.dist((self.state @ W, self.E).item()
        return loss

    
        

class TestTuner(StageTuner):
    def __init__(self, hyper_opts, logger,subPack, search_times):
        # self.xxxx = opts.xxxx
        # super().__init__(self.opts.tunings,search_times)
        super().__init__(hyper_opts,search_times)
        self.hyper_opts = hyper_opts
        self.logger = logger
        self.subPack = subPack        

        # self.tunings = opts.tunings
    
    def _fitness(self, config):

        # for key,value in enumerate(config.items()):
        #     setattr(self.hyper_opts.hyper,key,value)
        for key in config.keys():
            self.hyper_opts.hyper.dict[key] = config[key]
        # self.hyper_opts.hyper.hidden_size = config['hidden_size']
        # config has a dict type, as well as hyper
        # thus, this function can be transfered into the search_param. by checking the key in config.dict, and then change the value the key in hyper

        model = importlib.import_module(self.hyper_opts.import_path)
        model = getattr(model, self.hyper_opts.class_name)
        model = model(self.hyper_opts.hyper, self.logger)

        train_loader = self.subPack.train_loader
        valid_loader = self.subPack.valid_loader

        vmase = model.xfit(train_loader, valid_loader)

        score = vmase        
        
        return score