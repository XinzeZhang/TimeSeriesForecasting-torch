import os
import sys

from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error

import numpy as np
import torch.nn as nn
import torch

from task.util import os_makedirs
from task.util import plot_xfit

from tqdm import trange

import gc

def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    if model.params.device == torch.device('cpu'):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint, map_location='cuda')

    model.hidden_size = checkpoint['hidden_size']
    model.best_hidden_size = checkpoint['best_hidden_size']
    model.weight_IH = checkpoint['weight_IH']
    model.bias_IH = checkpoint['bias_IH']
    model.weight_HO = checkpoint['weight_HO']
    model.best_HO = checkpoint['best_HO']
    model.loss_list = checkpoint['loss_list']
    model.vloss_list = checkpoint['vloss_list']

def save_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(
        checkpoint, 'train.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)


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


class scnModel(nn.Module):
    '''
    Stochastic Configuration Networks
    '''

    def __init__(self,params=None, logger=None):
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        super(scnModel, self).__init__()
        self.input_dim = params.steps
        self.output_dim = params.H
        self.candidate_size = params.candidate_size
        self.hidden_size = params.hidden_size
        self.best_hidden_size = 1
        self.Lambdas = params.Lambdas
        self.r = params.r
        self.tolerance = params.tolerance


        with torch.no_grad():
            self.weight_IH = initWeight(self.input_dim, 1).to(self.params.device)
            self.bias_IH = initBiases().to(self.params.device)
        self.weight_HO = None
        self.weight_candidates = None
        self.bias_candidates = None
        # self.ridge_alpha = 0.1
        # self.regressor = Ridge(alpha=self.ridge_alpha)
        # self.best_regressor = Ridge(alpha=self.ridge_alpha)


        self.loss_list = []
        self.vloss_list = []
        self.params.plot_dir = os.path.join(params.task_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

        self.loss_fn = nn.MSELoss()
        
        self.best_state = None

    def load_state(self,):
        assert self.best_state is not None
        self.hidden_size = self.best_state['hidden_size']
        self.best_hidden_size = self.best_state['best_hidden_size']
        self.weight_IH = self.best_state['weight_IH']
        self.bias_IH = self.best_state['bias_IH']
        self.weight_HO = self.best_state['weight_HO']
        self.best_HO = self.best_state['best_HO']
        self.loss_list = self.best_state['loss_list']
        self.vloss_list = self.best_state['vloss_list']

    def construct(self, epoch, input, target, error):
        self.find = False
        self.best_idx = None
        self.error_dim = error.data.size(1)

        for Lambda in self.Lambdas:
            self.weight_candidates = torch.empty(
                self.input_dim, self.candidate_size).uniform_(-Lambda, Lambda).float().to(self.params.device)
            self.bias_candidates = torch.empty(
                1, self.candidate_size).uniform_(-Lambda, Lambda).float().to(self.params.device)

            temp1_array = []
            temp2_array = []
            scores = torch.zeros(self.candidate_size, len(self.r)).to(self.params.device)

            # shape: (N, candidate_size)
            candidates_state = input.mm(
                self.weight_candidates) + self.bias_candidates
            candidates_state = torch.sigmoid(candidates_state)

            for idx in range(self.candidate_size):
                c_idx = candidates_state[:, idx]
                c_idx = torch.reshape(
                    c_idx, (candidates_state.data.size(0), 1))  # shape :(N,1)

                left_vector = torch.zeros(self.error_dim)
                right_vector = torch.zeros(self.error_dim)

                for dim in range(self.error_dim):
                    e_dim = error[:, dim].reshape(-1, 1)

                    left_vector[dim] = (torch.pow(torch.mm(e_dim.t(), c_idx),
                                      2) / torch.mm(c_idx.t(), c_idx))[0]
                    right_vector[dim]= torch.mm(e_dim.t(), e_dim)[0,0]


                    for i, r_l in enumerate(self.r):
                        scores[idx, i] += left_vector[dim] - (1-r_l) * right_vector[dim]
            
            # max_idx = torch.argmax(scores)
            # max_idx = np.unravel_index(max_idx.cpu(), scores.shape)
            # best_c_idx = max_idx[0]
            # best_r_idx = max_idx[1]
            # best_r = r[best_r_idx]
            # best_score = scores[best_c_idx, best_r_idx]
            for i, r_l in enumerate(self.r):
                score_r = scores[:, i]
                if score_r.max() > 0:
                    best_c_idx = torch.argmax(score_r)
                    best_r = r_l
                    best_score = score_r.max()
                    
                    self.find = True
                    self.best_idx = best_c_idx
                    # self.best_r = best_r
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
        return H_state

    def solve_output(self, feature, target):
        output_w, _ = torch.lstsq(target, feature)
        output_w = output_w[0:feature.size(1)].to(self.params.device)
        return output_w

    def xfit(self, train_loader, val_loader, checkpoint = False):
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
        with torch.no_grad():
            min_vmse = 9999
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

            fit_error = train_y.clone().detach()

            v_loss_0 = 0
            for epoch in trange(1, self.hidden_size+1):
                if epoch >= 2:
                    if self.construct(epoch, train_x, train_y, fit_error) == False:
                        # once construct, hidden neurons add one
                        break

                H_state = self.forward(train_x)
                # solve the linear problem  H_state * Weight_HO = Output by ridge regression
                self.weight_HO = self.solve_output(H_state, train_y)
                # self.weight_HO, _ = torch.lstsq(train_y, H_state)
                pred = H_state.mm(self.weight_HO)
                fit_error = train_y - pred
                # solve the linear problem: H_state * Weight_HO = Output by least square
                # self.weight_HO, LU = torch.gesv(target,H_state) 
                # pred = torch.mm(H_state, self.weight_HO)
                # pred = pred.data.numpy()
                
                # save rmse as loss 
                loss = np.sqrt(self.loss_fn(pred, train_y).item())
                self.loss_list.append(loss)

                v_state = self.forward(val_x)
                vpred= v_state.mm(self.weight_HO)
                vloss = np.sqrt(self.loss_fn(vpred, val_y).item())
                if epoch == 1:
                    v_loss_0 = vloss
                else:
                    if vloss > v_loss_0:
                        vloss = v_loss_0
                
                self.vloss_list.append(vloss)
                self.logger.info('Hidden size: {} \t \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                    self.weight_IH.data.size(1), loss, vloss, min_vmse))

                if vloss < min_vmse:
                    min_vmse = vloss
                    self.best_hidden_size = self.weight_IH.data.size(1)
                    self.best_HO = self.weight_HO.detach().clone()
                    self.logger.info('Found new best state')
                    self.logger.info('Best vrmse: {:.4f}'.format(min_vmse))

                gc.collect()
        if checkpoint:
            save_checkpoint({
                        'hidden_size': self.weight_IH.data.size(1),
                        'best_hidden_size': self.best_hidden_size,
                        'best_HO': self.best_HO,
                        'cv': self.params.cv,
                        'weight_IH': self.weight_IH,
                        'bias_IH': self.bias_IH,
                        'weight_HO': self.weight_HO,
                        'loss_list': self.loss_list,
                        'vloss_list': self.vloss_list}, checkpoint=self.params.task_dir)
            self.logger.info(
                        'Checkpoint saved to {}'.format(self.params.task_dir))
        else:
            self.best_state = {
                        'hidden_size': self.weight_IH.data.size(1),
                        'best_hidden_size': self.best_hidden_size,
                        'best_HO': self.best_HO,
                        'cv': self.params.cv,
                        'weight_IH': self.weight_IH,
                        'bias_IH': self.bias_IH,
                        'weight_HO': self.weight_HO,
                        'loss_list': self.loss_list,
                        'vloss_list': self.vloss_list}

        plot_xfit(np.array(self.loss_list), np.array(self.vloss_list),
                    '{}_cv{}_loss'.format(self.params.dataset, self.params.cv), self.params.plot_dir)


    def predict(self, input, using_best = True):
        """Predict the output by SCN
        
        Parameters
        ----------
        input : torch array-like shape, (n_samples, Input_dim)
            The data to be transformed.
            
        Returns
        -------
        output : numpy array-like shape, (n_samples, Output_dim).
        """
        best_pth = os.path.join(self.params.task_dir,
                                'train.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth):
            self.logger.info(
                'Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self)
        elif self.best_state is not None:
            self.load_state()
            
        if using_best:
            self.weight_IH = self.weight_IH[:,:self.best_hidden_size].reshape(-1, self.best_hidden_size)
            self.bias_IH = self.bias_IH[:, :self.best_hidden_size].reshape(-1, self.best_hidden_size)
            self.weight_HO = self.best_HO.detach().clone()
        with torch.no_grad():
            input = torch.tensor(input).float().to(self.params.device)
            H_state = self.forward(input)
            pred = H_state.mm(self.weight_HO)
        return pred.cpu().numpy()
