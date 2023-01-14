import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from models.statistical.arima import arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
import numpy as np
from tqdm import trange

class HWSES(arima):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
    
    def gen_model(self, ):
        self.model = HW(self.history,seasonal_periods=self.opts.period ).fit()        
    
    def predict(self, input):
        H = self.opts.H
        
        test_samples = input.shape[0]
    
        yPred = np.empty((test_samples, H))
        for i in trange(test_samples):
            self.gen_model()
            i_pred = self.model.forecast(H)
            yPred[i,:]= i_pred - self.constant
            self.history = np.concatenate((self.history[1:], input[i,-1].reshape(-1,)), axis=0)
            self.history, self.constant = self.check_history()
        return yPred

          
    # def predict(self, input):
    #     H = self.opts.H
        
    #     test_samples = input.shape[0]
    #     pred_sec = H + input.shape[0] - 1
        
    #     if self.refit == False:
    #         self.gen_model()
            
    #         yPred = self.model.forecast(pred_sec) - self.constant
    #         yPred = create_dataset(yPred, H - 1)
            
    #     else:
    #         yPred = np.empty((test_samples, H))
    #         for i in trange(test_samples):
    #             self.model = HW(self.history, trend =  self.trend, seasonal=self.seasonal,seasonal_periods=self.period).fit()
                
    #             yPred[i,:]= self.model.forecast(H) - self.constant
    #             self.history = np.concatenate((self.history, input[i,-1].reshape(-1,)), axis=0)
    #             self.check_history()
    #             # print(history[-3:])
    #     return yPred