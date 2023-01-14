from models._comSetting import cnn_base, nn_base
from task.TaskLoader import Opt
from ray import tune

class esm_base(cnn_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ESM_CNN'
        self.innerTuning = True

    def hyper_modify(self):
        self.hyper.channel_size = 100
        self.hyper.candidate_size = 30
        self.hyper.hw_lambda = 0.5
        self.hyper.p_size = 3
        self.hyper.tolerance = 0
        self.hyper.nonlinearity = 'sigmoid'
        
    # def tuning_modify(self):
    #     self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
    #     self.tuning.p_size = tune.qrandint(2, 4, 1)

class eto_base(nn_base):
    '''
    default with bayes search
    '''
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/eto/wider/arch.py'
        self.class_name = 'ETO_SDNN'
        self.training = False
        self.arch = 'cnn'
        self.innerTuning = True
    
    def hyper_init(self):
        self.hyper.arch_choice = ['ces', 'esc','cnn', 'esn']
        self.hyper.patience = 5
        self.hyper.patience_bos = 20
        self.hyper.max_cells = 30
        self.hyper.input_dim = 1
        
        self.hyper.esn = nn_base()
        self.hyper.esn.name = 'esn'
        self.hyper.esn.hyper.hidden_size = 50
        self.hyper.esn.hyper.iw_bound = 0.1
        self.hyper.esn.hyper.hw_bound = 1.0
        self.hyper.esn.hyper.weight_scale = 0.9
        self.hyper.esn.hyper.nonlinearity = 'sigmoid'

        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 100, 5)
        self.hyper.esn.tuning.iw_bound = tune.loguniform(1e-5, 1e-1)
        self.hyper.esn.tuning.weight_scale = tune.uniform(0.2, 0.99)
        self.hyper.esn.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh'])
        self.hyper.esn.tuning.fc_io = tune.choice([True,False]) # attention! this settings should be false when using normal scaler in this models 

        self.hyper.cnn = nn_base()
        self.hyper.cnn.name = 'cnn'
        self.hyper.cnn.hyper.kernel_size = 6
        ## padding need be same for deeper model
        # self.hyper.cnn.hyper.padding = 'same'
        # self.hyper.cnn.hyper.padding_mode = 'circular'
        self.hyper.cnn.hyper.padding = 0
        self.hyper.cnn.hyper.pooling_size = 3
        self.hyper.cnn.hyper.pooling = True
        self.hyper.cnn.hyper.hw_bound = 0.5
        self.hyper.cnn.hyper.nonlinearity = 'sigmoid'

        self.hyper.cnn.tuning.kernel_size = tune.qrandint(2, 10, 1)
        self.hyper.cnn.tuning.hw_bound = tune.uniform(0.05, 0.99)
        self.hyper.cnn.tuning.pooling_size = tune.qrandint(2, 10, 1)
        self.hyper.cnn.tuning.pooling = tune.choice([True, False])
        
        # self.hyper.cnn.tuning.padding_mode = tune.choice(['zeros', 'reflect', 'replicate','circular']) # allow this search space for deeper models
        self.hyper.cnn.tuning.fc_io = tune.choice([True,False]) # attention! this settings should be false when using normal scaler in this models 
                
        self.hyper.mix = nn_base()
        self.hyper.mix.hyper.kernel_size = 17
        self.hyper.mix.hyper.padding = 0
        self.hyper.mix.hyper.pooling_size = 3
        self.hyper.mix.hyper.pooling = True
        self.hyper.mix.hyper.cnn_hw_bound = 0.5
        self.hyper.mix.hyper.esn_weight_scale = 0.9
        self.hyper.mix.hyper.esn_hidden_size = 100
        self.hyper.mix.hyper.esn_iw_bound = 0.1
        self.hyper.mix.hyper.esn_hw_bound = 0.5
        # self.hyper.mix.hyper.fc_io = True # this line should not be added in the hyper
        self.hyper.mix.hyper.nonlinearity = 'sigmoid'
        
    def hyper_modify(self):
        
        self.hyper.mix.tuning.kernel_size = self.hyper.cnn.tuning.kernel_size
        self.hyper.mix.tuning.pooling_size = self.hyper.cnn.tuning.pooling_size
        self.hyper.mix.tuning.cnn_hw_bound = self.hyper.cnn.tuning.hw_bound
        self.hyper.mix.tuning.esn_hidden_size = self.hyper.esn.tuning.hidden_size
        self.hyper.mix.tuning.esn_iw_bound = self.hyper.esn.tuning.iw_bound
        self.hyper.mix.tuning.esn_weight_scale = self.hyper.esn.tuning.weight_scale
        self.hyper.mix.tuning.fc_io = tune.choice([True, False])
        self.hyper.mix.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
         
        self.hyper.ces = nn_base()
        self.hyper.ces.name = 'ces'
        self.hyper.ces.update(self.hyper.mix)
        
        self.hyper.esc = nn_base()
        self.hyper.esc.name = 'esc'
        self.hyper.esc.update(self.hyper.mix)
        
        self.hyper.cnn.tuner.iters = 40 
        self.hyper.esn.tuner.iters = 40
        self.hyper.esc.tuner.iters = 40
        self.hyper.ces.tuner.iters = 40
        #for cell-training
        self.hyper.cTrain = True
        self.hyper.cTrain_info = Opt()
        self.hyper.cTrain_info.training_name = 'naive'
        self.hyper.cTrain_info.max_epoch = 50
        self.hyper.cTrain_info.metric_loss = 'training'
        # for readout-tuning
        self.hyper.rTune = True
        self.hyper.ho = Opt()
        self.hyper.ho.tuning = Opt()
        self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 10)
        self.hyper.ho.tuner = Opt()
        self.hyper.ho.tuner.iters = 40   
        self.hyper.ho.tuner.algo_name = 'tpe'
        
        for arch_name in self.hyper.arch_choice:
            self.hyper.dict[arch_name].tuner.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
            }
            self.hyper.dict[arch_name].tuner.algo_name = 'tpe'

# class eto_pcr_default(eto_base):
#     def ablation_modify(self):
#         # for pre-tuning
#         self.hyper.cnn.tuner.iters = 20 
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