import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import concatenate
import torch.nn.init as Init
from data_process._data_process import mape

class CNN(nn.Module):
    def __init__(self, in_channels=1,input_dim = 20, kernel_size=5, channel_size = 300,output_dim =1 ,Num_iters = 8000):
        super(CNN, self).__init__()
        self.In_channels = in_channels
        self.In_dim = input_dim
        self.Output_dim = output_dim
        # self.Output_dim = output_dim
        self.Channel_size = channel_size
        self.Kernel_size = kernel_size
        self.p_size = 3

        self.Conv = nn.Conv1d(self.In_channels,self.Channel_size,kernel_size,padding=0)
        # self.Conv.weight.data = Init.normal_(torch.empty(channel_size, in_channels, kernel_size).float(), std=0.015)
        # self.Conv.bias.data =  Init.normal_(torch.empty(channel_size).float(), std=0.015)
        self.fc  = nn.Linear(self.Channel_size * (self.In_dim - self.Kernel_size -self.p_size +2) ,self.Output_dim,bias=False)
        self.Pool = nn.AvgPool1d(kernel_size=self.p_size,stride=1,padding=0)

        self.Num_iters = Num_iters

        self.loss_list = []
        self.vloss_list =[]

    def load_state(self, file_path):
        state = np.load(file_path)
        conv1_W, conv1_B , fc_w = state['arr_0'], state['arr_1'], state['arr_2']
        self.Conv.weight.data = torch.from_numpy(conv1_W).cuda()
        self.Conv.bias.data = torch.from_numpy(conv1_B).cuda()
        # self.fc.weight.data = torch.from_numpy(fc_w).t()

    def forward(self,input):
        feature_map = self.Conv(input)
        feature_map = torch.sigmoid(feature_map)
        feature_map = self.Pool(feature_map)
        
        feature_map = feature_map.view(-1, self.Channel_size * (self.In_dim - self.Kernel_size -self.p_size +2))
        
        pred = self.fc(feature_map)

        return pred
    
    def fit_validate(self,x,y,vx,vy):
        x=x.cuda()
        y=y.cuda()
        vx = vx.cuda()
        vy=vy.cuda()
        metric = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.0015, momentum=0.99)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.95)

        train_rmse_array = np.empty(self.Num_iters)
        test_rmse_array = np.empty(self.Num_iters)
        train_pred_batch = np.empty([x.data.size(0),self.Output_dim,self.Num_iters])
        test_pred_batch = np.empty([vx.data.size(0),self.Output_dim,self.Num_iters])

        for iter in range(1, self.Num_iters +1):
            scheduler.step()
            pred = self.forward(x)
            loss = metric(pred, y)
            rmse = np.sqrt(loss.item())
            # self.loss_list.append(rmse)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vpred = self.forward(vx)
            vloss = metric(vpred, vy)
            vrmse = np.sqrt(vloss.item())
            # self.vloss_list.append(vrmse)

            train_rmse_array[iter-1] = rmse
            test_rmse_array[iter-1] = vrmse
            train_pred_batch[:,:,iter-1]=pred.cpu().data.numpy()
            test_pred_batch[:,:,iter-1]=vpred.cpu().data.numpy()

            if iter % 400 == 0:
                print('Channel size: %d \nTraining RMSE: %.8e \t Validating RMSE: %.8e' % (iter,rmse,vrmse))

        return train_rmse_array, test_rmse_array, train_pred_batch,test_pred_batch







    
