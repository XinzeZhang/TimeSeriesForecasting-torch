from task.TaskLoader import Opt
from ray import tune


class nn_base(Opt):
    def __init__(self):
        super().__init__()
        self.hyper = Opt()
        
        self.tuner = Opt()
        self.tuning = Opt()

        self.hyper_init()
        self.tuning_init()

        self.base_modify()
        self.hyper_modify()

        self.tuning_modify()
        self.ablation_modify()
        self.task_modify()
        
        self.common_process()

    def hyper_init(self,):
        pass

    def tuning_init(self,):
        # total cpu cores for tuning
        self.tuner.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
        }
        # gpu cards per trial in tune
        # tuner search times
        self.tuner.iters = 20
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
        # self.tuner.algo = 'rand'

    def base_modify(self,):
        pass

    def hyper_modify(self,):
        pass

    def tuning_modify(self):
        pass

    def ablation_modify(self):
        pass

    def task_modify(self):
        pass


    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')

class stat_base(nn_base):
    def hyper_init(self):
        self.arch = 'statistic'
        self.training = False

class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = True
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.hidden_size = 400
        self.hyper.epochs = 1000
        self.hyper.learning_rate = 0.01
        self.hyper.step_lr = 20

class cnn_base(nn_base):
    def __init__(self):
        # self.hyper = Opt()
        # self.tuning = Opt()
        super().__init__()
        self.training = False
        self.arch = 'cnn'

    def hyper_init(self):        
        self.hyper.channel_size = 100
        self.hyper.input_dim = 1
        self.hyper.kernel_size = 3
        self.hyper.p_size = 3
        self.hyper.hw_lambda = 0.5
        self.hyper.nonlinearity = 'sigmoid'                