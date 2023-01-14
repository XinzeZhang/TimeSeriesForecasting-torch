import numpy as np
from numpy.lib import RankWarning
from numpy.testing._private.utils import print_assert_equal
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(target, pred, ith=None):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    def h_mape(target, pred, ith):
        t, p = target[:, ith-1], pred[:, ith-1]
        
        mask = t != 0.0
        error_ith = (t[mask] - p[mask]) / t[mask]
        error_ith = np.fabs(error_ith)
        _e_h = error_ith.mean().item()
        return _e_h
        
    if ith is None:
        H = target.shape[1]
        errors = np.zeros(H)
        for h in range(H):
            errors[h] = h_mape(target, pred, h+1)
        output = errors.mean().item()
    else:
        output = h_mape(target, pred, ith)
    return output

def smape(target, pred, ith=None):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    def h_smape(target, pred, ith):
        t, p = target[:, ith-1], pred[:, ith-1]
        
        mask = t != 0.0
        error_ith = (t[mask] - p[mask]) / (t[mask] + p[mask])
        error_ith = np.fabs(error_ith)
        _e_h = error_ith.mean().item()
        return _e_h
        
    if ith is None:
        H = target.shape[1]
        errors = np.zeros(H)
        for h in range(H):
            _e_h = h_smape(target, pred, h+1)
            errors[h] = _e_h
        output = errors.mean().item()
    else:
        output = h_smape(target, pred, ith)
    return output

def rmse(target, pred, ith = None):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    if ith is None:
        t = target
        p = pred
    else:
        t = target[:, ith-1]
        p = pred[:, ith-1]
        
    return np.sqrt(mean_squared_error(t, p)).item()

def nrmse(target, pred, ith = None):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    \n
    reference: 
    @article{jaegerHarnessing2004,
    title = {Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication},
    shorttitle = {Harnessing Nonlinearity},
    author = {Jaeger, Herbert and Haas, Harald},
    year = {2004},
    month = apr,
    volume = {304},
    pages = {78--80},
    chapter = {Report},
    copyright = {American Association for the Advancement of Science},
    journal = {Science},
    number = {5667},
    pmid = {15064413}
    }
    '''
    
    if ith is None:
        t = target.reshape(-1,)
        p = pred.reshape(-1,)
    else:
        t = target[:,ith-1]
        p = pred[:, ith-1]
        
    
    mse = mean_squared_error(t,p)
    variance = np.var(t)
    if variance == 0: # for smoothing
        variance +=1
    output = np.sqrt(mse/variance).item()
    
    return output
    
    
def mase(target, pred, naiveP, ith = None):
    '''
    Mean absolute scaled error, ith belongs to [1, H] \n
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    
    reference:
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    '''
            
    def h_mase(target,pred, naiveP, ith):
        '''
        Computing the mase at i_th of the prediction horizon.
        '''
        t = target[:,ith-1] / naiveP
        p = pred[:, ith-1] / naiveP 
        
        # t = target[:,ith-1] 
        # p = pred[:, ith-1]  
        # _e = 0
        # for i in range(0, t.shape[0]-1):
        #     _e += abs(t[i+1]-t[i]) 
        # scale_error = _e /  (t.shape[0] - 1)
        # _h_mase = mean_absolute_error(t,p) / scale_error
            
        #     # Due to the multiple-step-ahead forecasting x_{t+1}, x_{t+2},...x_{t+h},, the naive forecasting using the last actual value x_t to make prediction, thus the prediction value of the naive method should be test_input[:, -1]
        # scale_error = _e /  (t.shape[0] - 1)
            
        _h_mase = mean_absolute_error(t,p) # question : should there be t[1:] and p[1:] for alignment?
        # mae = mean_absolute_error(t[1:],p[1:])
    
        return _h_mase
    
    if ith is None:
        H = target.shape[1]
        _h_mase  = np.zeros(H)
        for h in range(H):
            _h_mase[h] = h_mase(target,pred,naiveP,h+1)
        output = _h_mase.mean().item()
    else:
        output = h_mase(target,pred,ith)
    return output



if __name__ == "__main__":
    # y_true = np.array([[0, 1, 1.5], [0.5, 1, 1], [-1, 1, 1], [7, -6, -2]])
    # y_pred = np.array([[0, 1.01, 1], [0, 2, 1], [-1, 2, 1], [8, -5, -1]])
    # print(mape(y_true, y_pred))
    # print(smape(y_true, y_pred))

    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    print(mape(y_true, y_pred))
    print(smape(y_true, y_pred, 2))
    print(smape(y_true, y_pred))
    
    y_true = np.array([[1, 0.2], [1, 1]])
    y_pred = np.array([[0, 2], [-1, 2]])
    print(rmse(y_true, y_pred, ith=2))
    print(np.var(y_true))
    
    # y_true = np.array([[1, 1], [1, 1]]).reshape(-1,)
    # y_pred = np.array([[0, 2], [-1, 2]]).reshape(-1,)
    print(rmse(y_true, y_pred))
    print(np.var(y_true))
    
    print(nrmse(y_true, y_pred))
    print(mase(y_true, y_pred))