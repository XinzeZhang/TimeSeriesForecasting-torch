import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error
import numpy as np
# import time
import torch.nn as nn
import torch

class RVFL(nn.Module):
    '''
    RVFL network
    '''
    def __init__(self,params=None, logger=None):
        super(RVFL, self).__init__()
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.input_dim = params.steps
        self.output_dim = params.H
        self.hidden_size = params.hidden_size


        self.Lambdas = params.Lambdas

        Lambda = self.Lambdas[0]

        self.weight_IH = torch.empty(
            self.input_dim, self.hidden_size).uniform_(-Lambda, Lambda).float()
        self.bias_IH = torch.empty(
            1, self.hidden_size).uniform_(-Lambda, Lambda).float()
        self.weight_HO = None

        self.loss_fn = nn.MSELoss()

        self.params.device = 'cpu'

    def fc_transform(self, input):
        H = torch.mm(input, self.weight_IH) + self.bias_IH
        H = torch.sigmoid(H)
        H = torch.cat((input, H), 1)

        return H

    def forward(self, input):
        H = self.fc_transform(input)
        pred = H.mm(self.weight_HO)

        return pred

    def xfit(self, train_loader, val_loader,):
        with torch.no_grad():
            train_x, train_y = None, None
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()

            val_x, val_y = None, None
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.params.device)
                batch_y = batch_y.to(torch.float32).to(self.params.device)
                val_x, val_y = batch_x.detach().clone(), batch_y.detach().clone()
        
            H = self.fc_transform(train_x)

            output_w, _ = torch.lstsq(train_y, H)
            self.weight_HO = output_w[0:H.size(1)].to(self.params.device)

            pred = H.mm(self.weight_HO)

            loss = np.sqrt(self.loss_fn(pred, train_y).item())

            vpred= self.forward(val_x)
            vloss = np.sqrt(self.loss_fn(vpred, val_y).item())

            self.logger.info('Hidden size: {} \t \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} '.format(
                self.weight_IH.data.size(1), loss, vloss))

    def predict(self, input):
        with torch.no_grad():
            input = torch.tensor(input).float().to(self.params.device)
            pred  = self.forward(input)
        return pred.cpu().numpy()
