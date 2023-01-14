from models._comSetting import stat_base

class naiveL(stat_base):        
    def base_modify(self):
        self.import_path = 'models/statistical/naive.py'
        self.class_name = 'Naive'
    
    def hyper_modify(self):
        self.hyper.method = 'last'
        
class naiveA(naiveL):
    def hyper_modify(self):
        self.hyper.method = 'avg'
        
class naiveP(naiveL):
    def hyper_modify(self):
        self.hyper.method = 'period'  

class pdqArima(stat_base):        
    def base_modify(self):
        self.import_path = 'models/statistical/arima.py'
        self.class_name = 'pdqARIMA'
    def hyper_modify(self):
        self.hyper.refit = True 

class autoArima(stat_base):        
    def base_modify(self):
        self.import_path = 'models/statistical/arima.py'
        self.class_name = 'autoARIMA'
    def hyper_modify(self):
        self.hyper.refit = True         
        self.hyper.max_pdq = (32,1,10)
        self.hyper.max_PDQ = (30,1,10)
        self.hyper.max_iter = 30
        self.hyper.jobs = 1
        self.hyper.m = 1
        
class es(stat_base):
    def base_modify(self):
        self.import_path = 'models/statistical/hwses.py'
        self.class_name = 'HWSES'
    def hyper_modify(self):
        self.hyper.refit = True 

# class pdqarima(arima_base):        
#     def task_modify(self):
#         self.hyper.pdq = (30,2,2)
        
#         self.sid_hypers = [ Opt(self.hyper) for _ in ['Beijing', 'Chengdu', 'Guangzhou','Shanghai','Shenyang']]
#         self.sid_hypers[0].pdq = (30,2,7)
#         self.sid_hypers[1].pdq = (35,2,3)
#         self.sid_hypers[2].pdq = (32,2,2)
#         self.sid_hypers[3].pdq = (70,2,2)
#         self.sid_hypers[4].pdq = (20,1,19)