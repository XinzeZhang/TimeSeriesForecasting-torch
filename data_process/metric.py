import numpy as np
from sklearn.metrics import mean_squared_error

def mape(target, pred):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    n = target.shape[0]
    d = target.shape[1]

    errors = []
    for i in range(n):
        target_i = target[i, :]
        pred_i = pred[i, :]
        mask = target_i != 0.0
        error_i = (target_i[mask] - pred_i[mask])/target_i[mask]
        error_i = np.fabs(error_i)
        error_i = error_i.tolist()
        errors.extend(error_i)
    
    errors = np.array(errors)
    e = errors.mean().item()

    return e

def smape(target, pred):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    n = target.shape[0]
    d = target.shape[1]

    errors = []
    for i in range(n):
        target_i = target[i, :]
        pred_i = pred[i, :]
        mask = target_i != 0.0
        error_i = (target_i[mask] - pred_i[mask])/(target_i[mask] + pred_i[mask])
        error_i = np.fabs(error_i)
        error_i = error_i.tolist()
        errors.extend(error_i)
    
    errors = np.array(errors)
    e = errors.mean().item()

    return e

def rmse(target, pred):
    '''
    target : np.array(n_samples, n_outputs)\n
    pred   : np.array(n_samples, n_outputs)
    '''
    return np.sqrt(mean_squared_error(target, pred)).item()

if __name__ == "__main__":
    y_true = np.array([[0, 1, 1.5], [0.5, 1, 1], [-1, 1, 1], [7, -6, -2]])
    y_pred = np.array([[0, 1.01, 1], [0, 2, 1], [-1, 2, 1], [8, -5, -1]])
    print(mape(y_true, y_pred))
    print(smape(y_true, y_pred))

    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    print(mape(y_true, y_pred))
    print(smape(y_true, y_pred))