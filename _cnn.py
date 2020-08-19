from data_process.util import create_dataset, scale_raw, Params
import argparse
import torch
from models.Pretrain_CNN import CNN

from numpy import concatenate, atleast_2d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


# from models.DSCNN import DSCNN_v1


def toTorch(train_input, train_target, test_input, test_target):
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()
    return train_input, train_target, test_input, test_target


parser = argparse.ArgumentParser(
    description='Time Series Forecasting')
parser.add_argument('-step', type=int, default=1, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-dim', type=int, default=24)
parser.add_argument('-iters', type=int, default=16000)

if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join('models', 'gsCNN.params.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = Params(json_path)

    params.merge(args)

    #test
    params.dataset = 'MT_001'
    ts = np.load('./data/paper/{}.npy'.format(params.dataset))
    ts = ts.reshape(-1)
    # set_length = len(ts)
    segmentation = int(len(ts)*2/3)
    # np.savez('ts.npz',ts)
    # ts = ts.reshape(-1,1)

    params.steps=168
    params.H=24
    # test
    params.num_epochs = 30

    params.model_name = '{}_h{}_deepAR'.format(params.dataset,params.H)
    dataset = create_dataset(ts, look_back=params.steps + params.H - 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    X, Y = dataset[:, :(0 - params.H)], dataset[:, (0-params.H):]
    train_input = X[:segmentation, :]
    train_target = Y[:segmentation]
    test_input = X[segmentation:, :]
    test_target = Y[segmentation:]
  
    train_target = train_target.reshape(-1, params.H)
    test_target = test_target.reshape(-1, params.H)

    train_input, train_target, test_input, test_target = toTorch(
        train_input, train_target, test_input, test_target)

    train_input = torch.from_numpy(
        X[:segmentation, :].reshape(-1, 1, train_input.shape[1])).float()
    test_input = torch.from_numpy(
        X[segmentation:, :].reshape(-1, 1, test_input.shape[1])).float()


    train_rmse_batch = np.empty(10)
    test_rmse_batch = np.empty(10)
    train_pred_Batchs = np.empty((train_input.data.size(0),params.H,10))
    test_pred_Batchs = np.empty((test_input.data.size(0),params.H,10))
    
    train_rmse_loss_batch = np.empty((args.iters,10))
    test_rmse_loss_batch = np.empty((args.iters,10))


    for t in range(10):
        print("Training and testing trail times: "+str(t))

        train2, test2,  train_pred_batch, test_pred_batch = None , None ,None ,None

        min_r = 10

        if not os.path.exists('./State/H'+str(h)):
            os.makedirs('./State/H'+str(h))
        for time in range(3):
            cnn = CNN(
                channel_size=100,kernel_size=int(dim/4),input_dim = dim, output_dim=h,Num_iters = args.iters).cuda()

            _train2, _test2,  _train_pred_batch, _test_pred_batch = cnn.fit_validate(train_input, train_target, test_input, test_target)
            min_rmse = np.amin(_test2)

            if min_rmse < min_r:
                min_r = min_rmse
                train2, test2,  train_pred_batch, test_pred_batch = _train2, _test2,  _train_pred_batch, _test_pred_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_rmse_loss_batch[:,t] = train2
        train_rmse_batch[t] = np.amin(train2)
        train_idx = np.argmin(train2)

        test_rmse_loss_batch[:,t] = test2
        test_rmse_batch[t] = np.amin(test2)
        test_idx = np.argmin(test2)
        
        train_pred_Batchs[:,:,t] = train_pred_batch[:,:,train_idx]
        test_pred_Batchs[:,:,t] = test_pred_batch[:,:,test_idx]            

        print(args.dataset + '\t H: ' + str(h) + '\t RMSE: '+ str(min_r))

    if not os.path.exists('./Pred_mbatch/H'+str(h)+'/cnn'):
        os.makedirs('./Pred_mbatch/H'+str(h)+'/cnn')
    np.savez('./Pred_mbatch/H'+str(h)+'/cnn'+'/'+data+'-cnn-batch.npz',
             train_rmse_loss_batch, test_rmse_loss_batch, train_rmse_batch, test_rmse_batch, train_pred_Batchs, test_pred_Batchs,)
    print('\n------------------------------------------------')
    print('Saving ' + './Pred_mbatch/H'+str(h)+'/cnn'+'/'+data+'-cnn-batch.npz' + ' Successfully!')
    print('------------------------------------------------')            

 









    # train2, test2, tra_mape2, test_mape2 = None , None ,None ,None

    # min_r = 10
    # min_m = 10
    # for time in range(10):
    #     cnn = CNN(
    #         channel_size=100,kernel_size=int(dim/4),input_dim = dim, output_dim=h,Num_iters = 4000).cuda()
    #     # cnn.load_state('./State/H'+str(h)+'/'+data+'C'+str(2)+'_H'+str(h)+'.npz')

    #     _train2, _test2, _tra_mape2, _test_mape2 = cnn.fit_validate(
    #         train_input, train_target, test_input, test_target)
    #     min_rmse = min(_test2)
    #     if min_rmse < min_r:
    #         min_r = min_rmse
    #         train2, test2, =_train2, _test2
    #     min_mape = min(_test_mape2)
    #     if min_mape < min_m:
    #         min_m = min_mape
    #         tra_mape2, test_mape2 = _tra_mape2, _test_mape2
    # print('\n------------------------------------------------')
    # print('CNN Model finished fitting')
    # print('------------------------------------------------')
    # print(args.dataset + '\t H: ' + str(h) + '\t RMSE: '+ str(min_r) + '\t MAPE: '+ str(min_m))

    # if not os.path.exists('./Pred/H'+str(h)):
    #     os.makedirs('./Pred/H'+str(h))
    # np.savez('./Pred/H'+str(h)+'/'+data+'-cnn.npz',
    #         #  scn_train, scn_test, scn_tra_mape, scn_test_mape,
    #         #  train1, test1, tra_mape1, test_mape1)
    #          train2, test2, tra_mape2, test_mape2)
    #         #  train3, test3, tra_mape3, test_mape3)

