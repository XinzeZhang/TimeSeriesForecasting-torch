import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import concatenate
import os

def match_rmse(h, data_name,model_name):
    result_dir = "./Pred_mbatch/H"+ str(h) + "/"

    file_name = result_dir + model_name +'/'+ data_name + '-'+model_name+'-batch.npz'

    metrics = np.load(file_name)
    # training_rmse_loss_batch = metrics['arr_0'].mean(1)
    # testing_rmse_loss_batch = metrics['arr_1'].mean(1)
    # training_rmse_loss_batch = metrics['arr_0']
    test_rmse_batch = metrics['arr_3']

    column = []
    column.append(model_name)
    for i in test_rmse_batch.tolist():
        column.append(i)

    return column


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

def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] -1)

    for i in range(s.shape[0]):
        s[i]=np.diagonal(np.flip(a,1), offset= -i + h-1,axis1=0,axis2=1).copy().mean()
    
    return s

def mape(y_true, y_pred): 
    y_true = unpadding(y_true)
    y_pred = unpadding(y_pred)

    mask =  y_true != 0.0
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    N_metric =  (y_true[mask] - y_pred[mask])/y_true[mask]
    N_metric = np.fabs(N_metric)
    metric = N_metric.mean()

    return metric

def smape(y_true, y_pred): 
    y_true = unpadding(y_true)
    y_pred = unpadding(y_pred)

    mask =  y_true != 0.0
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    N_metric =  (y_true[mask] - y_pred[mask])/(y_true[mask] + y_pred[mask])
    N_metric = np.fabs(N_metric)
    metric = N_metric.mean()

    return metric


def get_testTarget(subdata_file_path,h):
    ts = np.load(subdata_file_path)
    ts = ts["arr_0"].reshape(-1)
    segmentation = int(len(ts)*2/3)
    # np.savez('ts.npz',ts)
    # ts = ts.reshape(-1,1)

    # scaler, ts_scaled = scale_raw(ts)
    scaler = MinMaxScaler(feature_range=(-1,1))

    input_dim = None
    data_list = ['AR1', 'Brent', 'stock', 'Web']
    dim_list = [15,24,20,30]
    for i in range(4):
        if data_list[i] in subdata_file_path:
            input_dim = dim_list[i]
    
    dataset = create_dataset(ts,look_back=input_dim + h -1)
    S_dataset = scaler.fit_transform(dataset)
    X,Y = dataset[:,:(0 - h)], dataset[:,(0-h):]
    train_input = X[:segmentation,:]
    train_target = Y[:segmentation]
    test_input=X[segmentation:,:]
    test_target=Y[segmentation:]
    # train_input, test_input, train_target, test_target = train_test_split(X, Y, test_size=.2, random_state=42)
    # train_target = train_target.reshape(-1)
    # test_target = test_target.reshape(-1)
    return test_target, scaler, test_input


def match(file_path,measure,subdata_file_path,h):

    test_target, scaler, test_input = get_testTarget(subdata_file_path,h) # shape N 1

    pred_data = np.load(file_path)
    
    metric_10times = np.empty(10)
    test_pred_batch = pred_data['arr_5'] # shape N 10
    test_rmse_batch = pred_data['arr_3']
    for i in range(10):
        test_pred = test_pred_batch[:,:,i]
        pre_inverted = concatenate((test_input,test_pred),axis = 1)
        test_pred = scaler.inverse_transform(pre_inverted)[:,(0-h):]
        if measure == 'mape':
            mape_i = mape(test_target, test_pred)
            metric_10times[i] = mape_i
        if measure == 'rmse':
            # metric_10times[i] = rmse(test_target,test_pred)
            metric_10times[i] = test_rmse_batch[i]
        if measure == 'smape':
            mape_i = smape(test_target,test_pred)
            metric_10times[i] = mape_i
    # mape_10times = pred_data['arr_3']
    return metric_10times


def match_metric(h, measure, dataset_file,result_file,model):
    metric_array = match(result_file,measure, dataset_file,h)
    row = []
    row.append(model)
    for i in metric_array.tolist():
        row.append(i)
    
    return row

for h in [1,3,6]:
    result_dir = "./Pred_mbatch/H"+ str(h) + "/"

    for data in ['AR1','Brent','Web']:

        input_dir = "./Data/paper/"
        for measure in ['mape','smape','rmse']:

            table = []
            index_list = []
            index_list.append('Dataset')
            for i in range(1,11):
                index_list.append(str(i))
            table.append(index_list)

            for model in ['cnn', 'scn','scnn','v1','v2']:

                dataset_file = input_dir + data +'.npz'
                result_file = result_dir + model +'/'+ data + '-'+model+'-batch.npz'

                table.append(match_metric(h, measure, dataset_file,result_file,model))
            
            table_T = list(map(list,zip(*table)))

            with open('./Multiple_test/'+data+'_'+measure+'_'+ str(h) +'.csv','w') as f:
                writer = csv.writer(f)
                writer.writerows(table_T)
                print('./Multiple_test/' +data+'_'+measure+'_'+ str(h) +'.csv'+ " Successfully!") 

# input_dir = "./Data/paper/"
# for h in [1,3,6]:
#     result_dir = "./Pred_mbatch/H"+ str(h) + "/"
#     for measure in ['mape','smape','rmse']:
#         table = []
#         index_list = []
#         index_list.append('Dataset')
#         for i in range(1,31):
#             index_list.append(str(i))
#         table.append(index_list)
#         for model in ['cnn', 'scn','scnn','v1','v2']:
#             m_list = []
#             m_list.append(model)
#             for data in ['AR1','Brent','Web']:
#                 dataset_file = input_dir + data +'.npz'
#                 result_file = result_dir + model +'/'+ data + '-'+model+'-batch.npz'

#                 m_list.extend(match_metric(h, measure, dataset_file,result_file,model))
            
#             table.append(m_list)
        
#         table_T = list(map(list,zip(*table)))

#         with open('./Multiple_test/'+measure +'_'+ str(h) +'.csv','w') as f:
#             writer = csv.writer(f)
#             writer.writerows(table_T)
#             print('./Multiple_test/'+measure +'_'+ str(h) +'.csv'+ " Successfully!")         

# input_dir = "./Data/paper/"

# for measure in ['mape','smape','rmse']:
#     table = []
#     index_list = []
#     index_list.append('Dataset')
#     for i in range(1,91):
#         index_list.append(str(i))
        
#     table.append(index_list)

#     for model in ['cnn', 'scn','scnn','v1','v2']:
#         m_list = []
#         m_list.append(model)
        
#         for h in [1,3,6]:
#         result_dir = "./Pred_mbatch/H"+ str(h) + "/"
#             for data in ['AR1','Brent','Web']:
#                 dataset_file = input_dir + data +'.npz'
#                 result_file = result_dir + model +'/'+ data + '-'+model+'-batch.npz'

#                 m_list.extend(match_metric(h, measure, dataset_file,result_file,model))
            
#         table.append(m_list)
    
#     table_T = list(map(list,zip(*table)))

#     with open('./Multiple_test/'  +measure+'.csv','w') as f:
#         writer = csv.writer(f)
#         writer.writerows(table_T)
#         print('./Multiple_test/' +measure+'.csv'+ " Successfully!") 