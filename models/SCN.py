from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import torch.nn as nn
import torch


from data_process._data_process import timeSince,plot_loss,plot_train

def initWeight(input_dim, hidden_size=1, grad=False):
    result = torch.empty(input_dim, hidden_size).float()
    result = nn.init.normal_(result, std=0.1)
    if grad == True:
        result.requires_grad = True
    return result


def initBiases(hidden_size=1, grad=False):
    result = torch.empty(1, hidden_size).float()
    result = nn.init.normal_(result, std=0.1)
    if grad == True:
        result.requires_grad = True
    return result

# ---------------
# Stochastic Configuration Networks
# ---------------


class scnModel():
    '''
    Stochastic Configuration Networks
    '''

    def __init__(self, Input_dim, Output_dim=1, Hidden_size=100, Candidate_size=100, Ridge_alpha = 0.1,print_interval=50, plot_interval=1, plot_=False):
        super(scnModel, self).__init__()
        self.input_dim = Input_dim
        self.output_dim = Output_dim
        self.candidate_size = Candidate_size
        self.hidden_size = Hidden_size
        self.Lambdas = [0.5, 1, 5, 10, 30, 50, 100, 150, 200, 250]
        self.r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
        self.tolerance = 0.001
        self.loss = 10
        self.weight_IH = initWeight(self.input_dim, 1)
        self.bias_IH = initBiases()
        self.weight_HO = None
        self.weight_candidates = None
        self.bias_candidates = None
        self.ridge_alpha = Ridge_alpha
        self.regressor = Ridge(alpha=self.ridge_alpha)

        self.Print_interval = print_interval
        self.Plot_interval = plot_interval
        self.plot_ = plot_

    def construct(self, epoch, input, target, error):
        self.find = False
        self.best_idx = None
        self.error_dim = error.data.size(1)

        for Lambda in self.Lambdas:
            self.weight_candidates = torch.empty(
                self.input_dim, self.candidate_size).uniform_(-Lambda, Lambda).float()
            self.bias_candidates = torch.empty(
                1, self.candidate_size).uniform_(-Lambda, Lambda).float()

            temp1_array = []
            temp2_array = []

            # shape: (N, candidate_size)
            candidates_state = input.mm(
                self.weight_candidates) + self.bias_candidates
            candidates_state = torch.sigmoid(candidates_state)

            for idx in range(self.candidate_size):
                c_idx = candidates_state[:, idx]
                c_idx = torch.reshape(
                    c_idx, (candidates_state.data.size(0), 1))  # shape :(N,1)

                for dim in range(self.error_dim):
                    e_dim = error[:, dim]
                    e_dim = torch.reshape(
                        e_dim, (error.data.size(0), 1))  # shape : (N, 1)

                    temp1 = torch.pow(torch.mm(e_dim.t(), c_idx),
                                      2) / torch.mm(c_idx.t(), c_idx)
                    temp2 = torch.mm(e_dim.t(), e_dim)
                    temp1_array.append(temp1)
                    temp2_array.append(temp2)

            for r_l in self.r:
                criteria_max = -0.1
                for idx in range(self.candidate_size):
                    criteria = temp1_array[idx] - \
                        (1-r_l) * temp2_array[idx]*epoch/(epoch+1)
                    criteria = criteria.data.numpy()
                    if criteria >= 0 and criteria > criteria_max:
                        criteria_max = criteria
                        self.best_idx = idx
                        self.find = True
                if self.find:
                    break

            if self.find:
                weight_new = torch.reshape(self.weight_candidates[:, self.best_idx], (
                    self.weight_candidates.data.size(0), 1))  # shape : (N,1)
                bias_new = torch.reshape(
                    self.bias_candidates[:, self.best_idx], (1, 1))  # shape : (1,1)
                # shape: (N, 1+epoch)
                self.weight_IH = torch.cat((self.weight_IH, weight_new), 1)
                # shape : (1, 1+epoch)
                self.bias_IH = torch.cat((self.bias_IH, bias_new), 1)
                break
        if self.find == False:
            print('End searching!')
            return False

        return True

    def forward(self, input):
        H_state = torch.mm(input, self.weight_IH) + self.bias_IH
        H_state = torch.sigmoid(H_state)
        H_state = H_state.data.numpy()

        return H_state

    def fit(self, input, target, save_road='./Results/'):
        """fit the data to scn
        
        Parameters
        ----------
        input : torch array-like shape, (n_samples, Input_dim)
            The data to be transformed.
        target : torch array-like shape, (n_samples, Output_dim)
            The data to be transformed.
            
        Returns
        -------
        self : returns an instance of self.
        """
        fit_error = target.clone().detach()
        epoch = 1
        # Initialize timer
        time_tr_start = time.time()
        if self.plot_ == True:
            plot_losses = []
        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        while (epoch<=self.hidden_size) and (self.loss > self.tolerance):

            if epoch >= 2:
                if self.construct(epoch, input, target, fit_error) == False:
                    # once construct, hidden neurons add one
                    break

            H_state = self.forward(input)

            # solve the linear problem  H_state * Weight_HO = Output by ridge regression
            self.regressor.fit(H_state, target.data.numpy())
            pred = self.regressor.predict(H_state)
            # solve the linear problem: H_state * Weight_HO = Output by least square
            # self.weight_HO, LU = torch.gesv(target,H_state) 
            # pred = torch.mm(H_state, self.weight_HO)
            # pred = pred.data.numpy()

            self.loss = mean_squared_error(target.data.numpy(), pred)

            training_rmse = np.sqrt(self.loss)
            if self.plot_ == True:
                train_plot_loss_total += training_rmse
            train_print_loss_total += training_rmse

            fit_error = torch.from_numpy(pred).float() - target

            if epoch % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0
                print('%s (%d %d%%) ' % (timeSince(time_tr_start, epoch / self.hidden_size),
                                             epoch, epoch / self.hidden_size * 100))
                print('Training RMSE:  \t %.3e' % (print_loss_avg))
            if self.plot_ == True:
                if epoch % self.Plot_interval == 0:
                    plot_loss_avg = train_plot_loss_total / self.Plot_interval
                    plot_losses.append(plot_loss_avg)
                    train_plot_loss_total = 0

            epoch = epoch+1
        if self.plot_ == True:
            plot_loss(plot_losses, Fig_name=save_road+'Loss_'+"SCN" + '_H' + str(self.hidden_size) +'_C' + str(self.candidate_size))
        print('\n------------------------------------------------')
        print('SCN Model finished fitting')
        print('------------------------------------------------')

    def predict(self, input):
        """Predict the output by SCN
        
        Parameters
        ----------
        input : torch array-like shape, (n_samples, Input_dim)
            The data to be transformed.
            
        Returns
        -------
        output : numpy array-like shape, (n_samples, Output_dim).
        """
        H_state = self.forward(input)
        pred = self.regressor.predict(H_state)
        return pred
