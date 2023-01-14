from models._comSetting import nn_base
from task.TaskLoader import Opt
from ray import tune

class esn_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'rnn'
        super().__init__()
        
    def base_modify(self,):
        self.import_path='models/stochastic/esn/ESN.py'
        self.class_name = 'EchoStateNetwork'
        
    def hyper_init(self,):        
        self.hyper.leaky_r = 1
        self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
        self.hyper.hidden_size = 400
        self.hyper.lambda_reg = 0
        self.hyper.nonlinearity = 'tanh'
        self.hyper.iw_bound = (-0.1, 0.1)
        self.hyper.hw_bound = (-1, 1)
        self.hyper.weight_scaling = 0.9
        self.hyper.init = 'vanilla'
        self.hyper.fc_io = 'step'
        self.hyper.input_dim = 1
    
    def tuning_init(self):
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        self.tuner.iters = 20
        self.tuning.iw_bound = tune.loguniform(1e-5, 1e-1)
        self.tuning.weight_scaling = tune.uniform(0.2, 0.99)
        self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh'])        
    
class ar_esn_base(esn_base):
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'SSO_ESN'

class rd_esn_base(esn_base):
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'S2S_ESN'

class sm_esn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
    
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/StripESN.py'
        self.class_name = 'StateSelection'
        
    def tuning_init(self):
        self.tuner.iters = 200
        self.tuner.algo = 'pso'
        self.tuner.resource = {
            'cpu': 5, 'gpu': 0.5
        }
        
class gesn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Growing_ESN'
    
    def hyper_modify(self):
        self.hyper.hidden_size = 48
        self.hyper.branch_size = 10
        self.hyper.weight_scaling = 0.9
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.nonlinearity = 'sigmoid'
        
class desn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/DeepESN.py'
        self.class_name = 'Deep_ESN'
    
    def hyper_modify(self):
        self.hyper.leaky_r = 0.55
        self.hyper.nonlinearity = 'tanh'
        self.hyper.hidden_size = 100
        self.hyper.num_layers = 10