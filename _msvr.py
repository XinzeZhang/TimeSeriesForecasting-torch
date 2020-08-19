from data_process.util import create_dataset ,de_scale , Params
import argparse
from models.MSVR import MSVR

from numpy import concatenate, atleast_2d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(
    description='Time Series Forecasting Baseline model')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-steps', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dim', type=int, default=24)
parser.add_argument('-k', type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()
    # json_path = os.path.join('models', 'deepAR.params.json')
    
    params = Params(json_path=None, zero_start=True)

    params.merge(args)
    
    #test
    params.dataset = 'AR1'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    # segmentation = int(len(ts)*5/6)
    params.steps=15
    params.H=3

    params.model_name = '{}_h{}_mSVR'.format(params.dataset,params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    kf = KFold(n_splits=params.k)
    kf.get_n_splits(dataset)

    cvs=[]
    for train_idx, test_idx in tqdm(kf.split(dataset)):
        train_data = dataset[train_idx]
        test_data = dataset[test_idx]

        train_input = train_data[:, :(0 - params.H)]
        train_target = train_data[:, (0-params.H):]
        test_input = test_data[:, :(0 - params.H)]
        test_target = test_data[:, (0-params.H):]

        train_target = train_target.reshape(-1, params.H)
        test_target = test_target.reshape(-1, params.H)

        msvr = MSVR(epsilon=0.01, C=1000, gamma=0.01, tol=0.0001)
        rmse, vrmse, pred, vpred = msvr.fit_validate(
            train_input, train_target, test_input, test_target)

        pred= msvr.predict(test_input)
        pred = de_scale(params,scaler,pred)
        target = de_scale(params,scaler,test_target)
        cvs.append((pred,target))

    measures = np.zeros(len(cvs))
    for i, (pred, target) in enumerate(cvs):
        vrmse = np.sqrt(mean_squared_error(target,pred))
        measures[i] = vrmse
        print('{}\t H: {}\t CV: {}\t RMSE:{}'.format(params.dataset, params.H, i, vrmse))
    print(params.dataset + '\t H: ' + str(params.H) + '\t Avg RMSE: ' + str(measures.mean()) + '\t')
