# # from ray import tune
# from task.TaskLoader import Opt
# from ray import tune

# class nn_base(Opt):
#     def __init__(self):
#         super().__init__()
#         self.hyper = Opt()
        
#         self.tuner = Opt()
#         self.tuning = Opt()

#         self.hyper_init()
#         self.tuning_init()

#         self.base_modify()
#         self.hyper_modify()

#         self.tuning_modify()
#         self.ablation_modify()
        
        
#         self.common_process()

#     def hyper_init(self,):
#         pass

#     def tuning_init(self,):
#         # total cpu cores for tuning
#         self.tuner.resource = {
#             "cpu": 10,
#             "gpu": 1  # set this for GPUs
#         }
#         # gpu cards per trial in tune
#         # tuner search times
#         self.tuner.iters = 20
#         # fitness epoch per iter
#         self.tuner.epochPerIter = 1
#         # self.tuner.algo = 'rand'

#     def base_modify(self,):
#         pass

#     def hyper_modify(self,):
#         pass

#     def tuning_modify(self):
#         pass

#     def ablation_modify(self):
#         pass

#     def common_process(self,):
#         if "import_path" in self.dict:
#             self.import_path = self.import_path.replace(
#             '.py', '').replace('/', '.')

# class stat_base(nn_base):
#     def hyper_init(self):
#         self.arch = 'statistic'
#         self.training = False


# class naiveL(stat_base):        
#     def base_modify(self):
#         self.import_path = 'models/statistical/naive.py'
#         self.class_name = 'Naive'
    
#     def hyper_modify(self):
#         self.hyper.method = 'last'
        
# class naiveA(naiveL):
#     def hyper_modify(self):
#         self.hyper.method = 'avg'
        
# class naiveP(naiveL):
#     def hyper_modify(self):
#         self.hyper.method = 'period'  

# class msvr_base(nn_base):
#     def base_modify(self):
#         self.import_path = 'models/stochastic/mlp/MSVR.py'
#         self.class_name = 'MSVR'
        
#     def hyper_init(self):
#         self.arch = 'statistic'
#         self.hyper.C = 1.0
#         self.hyper.kernel = 'rbf'
#         self.hyper.gamma = None
#         self.hyper.coef = 0.0
#         self.hyper.tol = 0.0001
#         self.hyper.epsilon = 0.001
#         self.hyper.degree = 3
        
# class mlp(nn_base):
#     def base_modify(self):
#         self.import_path = 'models/training/MLP.py'
#         self.class_name = 'MLP'
        
#         self.training = True
#         self.arch = 'mlp'
    
#     def hyper_init(self):
#         self.hyper.hidden_size = 400
#         self.hyper.epochs = 1000
#         self.hyper.learning_rate = 0.01
#         self.hyper.step_lr = 20

# class esn_base(nn_base):
#     def __init__(self):
#         self.training = False
#         self.arch = 'rnn'
#         super().__init__()
        
#     def base_modify(self,):
#         self.import_path='models/stochastic/esn/ESN.py'
#         self.class_name = 'EchoStateNetwork'
        
#     def hyper_init(self,):        
#         self.hyper.leaky_r = 1
#         self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
#         self.hyper.hidden_size = 400
#         self.hyper.lambda_reg = 0
#         self.hyper.nonlinearity = 'tanh'
#         self.hyper.iw_bound = (-0.1, 0.1)
#         self.hyper.hw_bound = (-1, 1)
#         self.hyper.weight_scaling = 0.9
#         self.hyper.init = 'vanilla'
#         self.hyper.fc_io = 'step'
#         self.hyper.input_dim = 1

# class sso_esn_base(esn_base):
#     def __init__(self):
#         super().__init__()

#     def base_modify(self,):
#         self.import_path = 'models/stochastic/esn/ESN.py'
#         self.class_name = 'SSO_ESN'

# class s2s_esn_base(esn_base):
#     def base_modify(self,):
#         self.import_path = 'models/stochastic/esn/ESN.py'
#         self.class_name = 'S2S_ESN'

# class ss_esn_base(nn_base):
#     def __init__(self):
#         super().__init__()
#         self.training = False
#         self.arch = 'rnn'
#         self.innerTuning = True
    
#     def base_modify(self,):
#         self.import_path = 'models/stochastic/esn/StripESN.py'
#         self.class_name = 'StateSelection'
        
#     def tuning_init(self):
#         self.tuner.iters = 200
#         self.tuner.algo = 'pso'
#         self.tuner.resource = {
#             'cpu': 5, 'gpu': 0.5
#         }
        
# class cnn_base(nn_base):
#     def __init__(self):
#         # self.hyper = Opt()
#         # self.tuning = Opt()
#         super().__init__()
#         self.training = False
#         self.arch = 'cnn'

#     def hyper_init(self):        
#         self.hyper.channel_size = 100
#         self.hyper.input_dim = 1
#         self.hyper.kernel_size = 3
#         self.hyper.p_size = 3
#         self.hyper.hw_lambda = 0.5
#         self.hyper.nonlinearity = 'sigmoid'

# class gsc_base(cnn_base):
#     def base_modify(self):
#         self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
#         self.class_name = 'ES_CNN'
    
#     def hyper_init(self):
#         self.hyper.input_dim = 1
#         self.hyper.channel_size = 100
#         self.hyper.kernel_size = 3
#         self.hyper.p_size = 3
#         self.hyper.hw_lambda = 0.5
#         self.hyper.nonlinearity = 'sigmoid'
        
# class esm_base(cnn_base):
#     def __init__(self):
#         super().__init__()

#     def base_modify(self):
#         self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
#         self.class_name = 'ESM_CNN'

#     def hyper_modify(self):
#         self.hyper.channel_size = 100
#         self.hyper.candidate_size = 30
#         self.hyper.hw_lambda = 0.5
#         self.hyper.p_size = 3
#         self.hyper.tolerance = 0
#         self.hyper.nonlinearity = 'sigmoid'
        
#     def tuning_modify(self):
#         self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
#         self.tuning.p_size = tune.qrandint(2, 4, 1)
#         # self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])


# class ice_base(nn_base):
#     def __init__(self):
#         # self.hyper = Opt()
#         # self.tuning = Opt()
#         super().__init__()
#         self.training = False
#         self.arch = 'cnn'
#         self.innerTuning = True

#     def hyper_init(self):
#         self.hyper.arch_choice = ['ces', 'esc','cnn', 'esn']
#         self.hyper.patience = 5
#         self.hyper.patience_bos = 20
#         self.hyper.max_cells = 30
        
#         self.hyper.esn = nn_base()
#         self.hyper.esn.name = 'esn'
#         self.hyper.esn.hyper.hidden_size = 50
#         self.hyper.esn.hyper.iw_bound = 0.1
#         self.hyper.esn.hyper.hw_bound = 1.0
#         self.hyper.esn.hyper.weight_scale = 0.9
#         self.hyper.esn.hyper.nonlinearity = 'sigmoid'

        
#         self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 100, 5)
#         self.hyper.esn.tuning.iw_bound = tune.uniform(0.1, 0.99)
#         self.hyper.esn.tuning.hw_bound = tune.uniform(0.1, 0.99)
#         self.hyper.esn.tuning.fc_io = tune.choice([True])  # attention! this settings should be false when using normal scaler in this models 

#         self.hyper.cnn = nn_base()
#         self.hyper.cnn.name = 'cnn'
#         self.hyper.cnn.hyper.kernel_size = 6
#         ## padding need be same for deeper model
#         # self.hyper.cnn.hyper.padding = 'same'
#         # self.hyper.cnn.hyper.padding_mode = 'circular'
#         self.hyper.cnn.hyper.padding = 0
#         self.hyper.cnn.hyper.pooling_size = 3
#         self.hyper.cnn.hyper.pooling = True
#         self.hyper.cnn.hyper.hw_bound = 0.5
#         self.hyper.cnn.hyper.nonlinearity = 'sigmoid'


#         self.hyper.cnn.tuning.kernel_size = tune.qrandint(2, 10, 1)
#         self.hyper.cnn.tuning.hw_bound = tune.uniform(0.05, 0.99)
#         self.hyper.cnn.tuning.pooling_size = tune.qrandint(2, 10, 1)
#         self.hyper.cnn.tuning.pooling = tune.choice([True, False])
        
#         # self.hyper.cnn.tuning.padding_mode = tune.choice(['zeros', 'reflect', 'replicate','circular']) # allow this search space for deeper models
#         self.hyper.cnn.tuning.fc_io = tune.choice([True]) # attention! this settings should be false when using normal scaler in this models 
                
#         self.hyper.mix = nn_base()
#         self.hyper.mix.hyper.kernel_size = 17
#         self.hyper.mix.hyper.padding = 0
#         self.hyper.mix.hyper.pooling_size = 3
#         self.hyper.mix.hyper.pooling = True
#         self.hyper.mix.hyper.cnn_hw_bound = 0.5
#         self.hyper.mix.hyper.esn_weight_scale = 0.9
#         self.hyper.mix.hyper.esn_hidden_size = 100
#         self.hyper.mix.hyper.esn_iw_bound = 0.1
#         self.hyper.mix.hyper.esn_hw_bound = 0.5
#         # self.hyper.mix.hyper.fc_io = True # this line should not be added in the hyper
#         self.hyper.mix.hyper.nonlinearity = 'sigmoid'
        
#     def tuning_modify(self):
        
#         self.hyper.mix.tuning.kernel_size = self.hyper.cnn.tuning.kernel_size
#         self.hyper.mix.tuning.pooling_size = self.hyper.cnn.tuning.pooling_size
#         self.hyper.mix.tuning.cnn_hw_bound = self.hyper.cnn.tuning.hw_bound
#         self.hyper.mix.tuning.esn_hidden_size = self.hyper.esn.tuning.hidden_size
#         self.hyper.mix.tuning.esn_iw_bound = self.hyper.esn.tuning.iw_bound
#         self.hyper.mix.tuning.esn_hw_bound = self.hyper.esn.tuning.hw_bound
#         self.hyper.mix.tuning.fc_io = tune.choice([True])
#         self.hyper.mix.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
         
#         self.hyper.ces = nn_base()
#         self.hyper.ces.name = 'ces'
#         self.hyper.ces.update(self.hyper.mix)
        
#         self.hyper.esc = nn_base()
#         self.hyper.esc.name = 'esc'
#         self.hyper.esc.update(self.hyper.mix)
        
# class wider_default(ice_base):
#     def base_modify(self):
#         self.import_path = 'models/stochastic/cnn/ice/wider/arch.py'
#         self.class_name = 'ICESN'
    
#     # def hyper_modify(self):
#     #     self.hyper.patience_bos = 100
#     #     self.hyper.max_cells = 150
                        
#     #     self.hyper.cnn.hyper.kernel_size = 4
                     
#     #     # self.hyper.esn.tuning.fc_io = tune.choice([True])   # default true
#     #     self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
#     #     self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        
#     def ablation_modify(self):
#         self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
#         self.hyper.esn.tuner.iters = 1
#         self.hyper.esc.tuner.iters = 1
#         self.hyper.ces.tuner.iters = 1
        

# class wider_pcr_default(wider_default):
#     def ablation_modify(self):
#         # for pre-tuning
#         self.hyper.cnn.tuner.iters = 16 
#         self.hyper.esn.tuner.iters = 16
#         self.hyper.esc.tuner.iters = 16
#         self.hyper.ces.tuner.iters = 16
#         #for cell-training
#         self.hyper.cTrain = True
#         self.hyper.cTrain_info = Opt()
#         self.hyper.cTrain_info.training_name = 'naive'
#         self.hyper.cTrain_info.max_epoch = 200
#         # for readout-tuning
#         self.hyper.rTune = True
#         self.hyper.ho = Opt()
#         self.hyper.ho.tuning = Opt()
#         self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 10)
#         self.hyper.ho.tuner = Opt()
#         self.hyper.ho.tuner.iters = 20
