import numpy as np
import os

import matplotlib
matplotlib.use('agg') # avoiding Invalid DISPLAY variable
import matplotlib.ticker as ticker
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_dir = "./Data/Crude_Oil_Price/ED_12/"
pred_dirs = "./Results/COP/ED_12/"

pred_dir_list = os.listdir(pred_dirs)
pred_dir_list.sort()

bench_list = [256, 512, 768, 1024]

def load_file(raw_dir,result_dir, cell , size):
    if cell == 'SVR':
        temp = np.load(result_dir+'/Npz_'+'SVR_'+'RBF'+'.npz')
    else:
        temp = np.load(result_dir+'/Npz_'+cell+'_L1_H'+str(size)+'_E5000_SGD.npz')
    
    train = temp["arr_0"]
    train = train.flatten().tolist()
    test = temp["arr_1"]
    test =test.flatten().tolist()

    data = np.load(raw_dir+"/dataSet.npz")
    raw_train, raw_test = data["arr_0"],data["arr_1"]
    train_target = raw_train[:,-1]
    test.insert(0,train_target[-1])

    return train,test

def pred_plot(train_pred, test_pred,cell, color):
    plt.plot(train_section, train_pred, color=color, linestyle= '-', label=cell + ' Training', linewidth=1)
    plt.plot(test_section, test_pred, color=color, linestyle= '-.', label=cell + ' Test', linewidth=1)

for  folder in list(filter(lambda x: 'WTI' in x, pred_dir_list)):
    print('\n------------------------------------------------')
    print('Loading Data: '+folder)

    raw_dir = input_dir+folder
    raw = np.load(raw_dir+"/rawSet.npz")
    raw=raw["arr_0"]
    raw_T=raw.shape[0]
    raw_section=[*range(raw_T)]
    raw_values=raw.tolist()

    idx=np.load(raw_dir+"/idxSet.npz")
    train_idx, test_idx=idx["arr_0"],idx["arr_1"]

    # --
    train_section = train_idx[:,-1].flatten().tolist()
    # --
    test_section = test_idx[:,-1].flatten().tolist()
    test_section.insert(0,train_section[-1])

    result_dir = pred_dirs+folder
    

    for size in bench_list:
        gru_train, gru_test =load_file(raw_dir,result_dir,'GRU',size)
        lstm_train, lstm_test = load_file(raw_dir,result_dir,'LSTM',size)
        rnn_train, rnn_test = load_file(raw_dir,result_dir,'RNN',size)
        mlp_train, mlp_test = load_file(raw_dir,result_dir,'Linear',size)
        svr_train, svr_test = load_file(raw_dir,result_dir,'SVR',size)

        # ====plot the figure=====
        plt.figure(figsize=(20, 5))
        # plt.title(
        #     'Forecasting Future Values for Time Series', fontsize=12)
        # plt.title('RMSE of Prediction: %(rmse).3e' %
        #           {'rmse': RMSE_pred}, fontsize=10)
        plt.xlabel('Input Sequence', fontsize=10)
        plt.ylabel('Value', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.plot(raw_section,raw_values,'k-',label='Raw Series', linewidth=1)
        # --svr
        pred_plot(svr_train,svr_test,'SVR', '#1f77b4')
        # --mlp
        pred_plot(mlp_train,mlp_test,'MLP','#ff7f0e')
        # --rnn
        pred_plot(rnn_train,rnn_test,'RNN','#2ca02c')
        # --gru
        pred_plot(gru_train,gru_test,'GRU','#d62728')
        # --lstm
        pred_plot(lstm_train,lstm_test,'LSTM','#9467bd')              
        plt.legend(loc='upper right')
        plt.savefig(result_dir+'/NN_' + str(size) + '.png')
        plt.close()
        print('Done!')