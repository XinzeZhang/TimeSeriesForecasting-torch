from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import math

import matplotlib
matplotlib.use('TkAgg') # avoiding Invalid DISPLAY variable; this annotation must be before 'import matplotlib.pyplot as plt'
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import time

from matplotlib import animation

import torch


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


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

# create a differenced series

def mape(y_true, y_pred): 
    mask =  y_true != 0.0
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return (np.fabs((y_true[mask] - y_pred[mask])/y_true[mask])).mean()

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff).values

# invert differenced value


def inverse_difference(hvalues, yhat, interval=1):
    ori = list()
    for i in range(len(yhat)):
        value = yhat[i] + history[-interval + i]
        ori.append(value)
    return Series(ori).values

def inverse_array(array_in):
    data=array_in.copy()
    data_idx = np.arange(len(array_in)-1,-1,-1)
    data_inverse = data[data_idx]
    return data_inverse

# scale train and test data to [-1, 1]
def scale_raw(raw):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # raw = raw.reshape(raw.shape[0], 1)
    scaler = scaler.fit(raw)
    # transform train
    norm_raw = scaler.transform(raw)
    norm_raw = norm_raw[:,0]
    return scaler, norm_raw

def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # train_scaled = train_scaled.astype(np.float32)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    # test_scaled = test_scaled.astype(np.float32)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value


def invert_scale(scaler, ori_array, pred_array):
    # reshape the array to 2D
    pred_array = pred_array.reshape(pred_array.shape[0], 1)
    ori_array = ori_array.reshape(ori_array.shape[0], ori_array.shape[1])
    # maintain the broadcast shape with scaler
    pre_inverted = concatenate((ori_array, pred_array), axis=1)
    inverted = scaler.inverse_transform(pre_inverted)
    # extraction the pred_array_inverted
    pred_array_inverted = inverted[:, -1]
    return pred_array_inverted

# invert differenced train value


def inverse_train_difference(history, y_train_prediction, look_back):
    ori = list()
    # # appended the base
    # for i in range(look_back+1):
    #     ori.append(history[i])
    # appended the inverted diff
    for i in range(len(y_train_prediction)):
        value = y_train_prediction[i] + history[look_back + i]
        ori.append(value)
    return Series(ori).values

# invert differenced value


def inverse_test_difference(history, Y_test_prediction, train_size, look_back):
    ori = list()
    for i in range(len(Y_test_prediction)):
        value = Y_test_prediction[i] + history[train_size + look_back + i]
        ori.append(value)
    return Series(ori).values


def plot_result(Test_datasets, Train_pred, Test_pred, Loss_pred, Fig_name='Prediction'):
    # get length from time-sequence
    ts_size = len(Test_datasets)
    train_size = len(Train_pred)
    test_size = len(Test_pred)
    look_back = ts_size - train_size - test_size - 1

    time_period = np.arange(ts_size)
    incept_scope = np.array(look_back + 1)
    train_scope = np.arange(look_back + 1, train_size + look_back + 1)
    test_scope = np.arange(train_size + look_back + 1, ts_size)

    plt.figure(figsize=(35, 5))
    plt.title(
        'Predict future values for time sequences\n(bule lines are predicted values)', fontsize=12)
    plt.title('MSE of Prediction: %(mse).3e' %
              {'mse': Loss_pred}, loc='right', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.plot(time_period, Test_datasets, 'r-', label='Original', linewidth=1)
    plt.plot(train_scope, Train_pred, 'g-', label='train', linewidth=1)
    plt.plot(test_scope, Test_pred, 'b-', label='prediction', linewidth=1)

    plt.legend(loc='upper right')
    plt.savefig(Fig_name + '.png')
    # plt.show()


def plot_regression_result(Train_target,Train_pred,Test_target, Test_pred, Loss_pred, Fig_name='Prediction'):
    # get length from time-sequence
    test_size = len(Test_target)
    train_size = len(Train_pred)

    test_scope = np.arange(test_size)

    train_scope = np.arange(train_size)


    plt.figure(figsize=(20, 5))
    plt.title(
        'Regression Future Values for Time Series', fontsize=12)
    plt.title('RMSE of Prediction: %(rmse).3e' %
              {'rmse': Loss_pred}, loc='right', fontsize=10)
    plt.xlabel('Input Sequence', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # plt.plot(train_scope, Train_target, 'c-', label='Training Target', linewidth=1)
    # plt.plot(train_scope, Train_pred, 'm-', label='Training Result', linewidth=1)
    plt.plot(test_scope, Test_target, 'b-', label='Test Target', linewidth=1)
    plt.plot(test_scope, Test_pred, 'r-', label='Test Result', linewidth=1)

    plt.legend(loc='upper right')
    plt.savefig(Fig_name + '.png')

def plot_forecasting_result(Train_target,Train_pred,Train_section,Test_target, Test_pred,Test_section, Loss_pred, Fig_name='Prediction'):
    # get section of training and test sequence   
    test_scope = Test_section
    train_scope = Train_section


    plt.figure(figsize=(20, 5))
    plt.title(
        'Forecasting Future Values for Time Series', fontsize=12)
    plt.title('RMSE of Prediction: %(rmse).3e' %
              {'rmse': Loss_pred}, loc='right', fontsize=10)
    plt.xlabel('Input Sequence', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.plot(train_scope, Train_target, 'c-', label='Training Target', linewidth=1)
    plt.plot(train_scope, Train_pred, 'm-', label='Training Result', linewidth=1)
    plt.plot(test_scope, Test_target, 'b-.', label='Test Target', linewidth=1)
    plt.plot(test_scope, Test_pred, 'r-.', label='Test Result', linewidth=1)

    plt.legend(loc='upper right')
    plt.savefig(Fig_name + '.png')
# time-transform


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# time-count


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# show loss


def plot_loss(points, Fig_name):
    plt.figure(figsize=(10, 5))
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.title('Loss of %(fig_name)s \n(Final Training RMSE:%(loss).3e)' %
              {'fig_name': Fig_name, 'loss': points[-1]}, fontsize=12)
    plt.savefig(Fig_name + '.png')
    plt.close()

def plot_train(train_losses,validate_losses,Fig_title,Fig_name):
    plt.figure(figsize=(5,5))
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(train_losses, 'g-', label= 'Training Loss')
    plt.plot(validate_losses, 'b-', label='Validating Loss')
    plt.legend(loc='upper right')
    plt.title('Loss of %(fig_title)s \n' %
              {'fig_title': Fig_title}, fontsize=12)
    plt.title('Final Training RMSE:%(train_loss).3e' %
              {'train_loss':  train_losses[-1]}, loc='right', fontsize=10)
    plt.xlabel('Iteration Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(Fig_name + '.png')
    plt.close()

# show gpu train


def visual_train(Figure_size, target, Viewlist, View_interval):
    Num_iters = len(Viewlist)
    target_size = target.size(0)  # continuously plot
    time_period = np.arange(target_size)  # continuously plot
    fig = plt.figure(figsize=(Figure_size[0], Figure_size[1]))
    ax = plt.subplot(111)

    lines = []
    target_view = target[:, -1].data.numpy().flatten()
    line, = ax.plot(time_period, target_view, 'r-',
                    label='Target', linewidth=2)
    lines.append(line)
    line, = ax.plot(time_period, np.linspace(
        0, 0, num=target_size), 'g-', label='Train Result', linewidth=2)
    lines.append(line)
    ax.set_ylim(-1, 1)

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    text_template = 'Iter = %s'
    text_iter = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def update(iter):
        # update data
        # for iter in range(1,self.Num_iters+1):

        prediction_view = Viewlist[iter].numpy().flatten()

        lines[1].set_ydata(prediction_view)

        text_iter.set_text(text_template % str((iter + 1) * View_interval))

        return tuple(lines) + (text_iter,)

    anim = animation.FuncAnimation(
        fig, update, frames=Num_iters, interval=20, blit=True, repeat=False)
    plt.legend(loc='upper right')
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

    anim.save('train.mp4', writer=writer)
    # plt.show()

def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] -1)

    for i in range(s.shape[0]):
        s[i]=np.diagonal(np.flip(a,1), offset= -i + h-1,axis1=0,axis2=1).copy().mean()
    
    return s