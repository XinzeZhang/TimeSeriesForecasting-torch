import os
import sys
from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import numpy as np
import torch.nn as nn
import torch

from task.util import os_makedirs, os_rmdirs
from task.util import plot_xfit, savebest_checkpoint

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

def save_checkpoint(state, checkpoint, cv):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(
        checkpoint, 'train.cv{}.pth.tar'.format(cv))
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


class IELM(nn.Module):
    '''
    Stochastic Configuration Networks
    '''

    def __init__(self,params=None, logger=None):
        self.params = params
        self.logger = logger
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        super(IELM, self).__init__()
        self.input_dim = params.steps
        self.output_dim = params.H
        self.candidate_size = params.candidate_size
        self.hidden_size = params.hidden_size
        self.best_hidden_size = 1
        self.Lambdas = params.Lambdas
        self.r = params.r
        self.tolerance = params.tolerance

        self.hidden = 1

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

    def forward(self, input, size):
        IH_w = self.weight_IH[:, :size].reshape(self.weight_IH.data.size(0), size)
        IH_b = self.bias_IH[:, :size].reshape(self.bias_IH.data.size(0), size)
        HO_w = self.weight_HO[:size, :].reshape(size, self.weight_HO.data.size(1))
        H_state = torch.mm(input, IH_w) + IH_b
        H_state = torch.sigmoid(H_state)
        pred = torch.mm(H_state, HO_w)
        return pred

    def solve_output(self, feature, target):
        output_w, _ = torch.lstsq(target, feature)
        output_w = output_w[0:feature.size(1)].to(self.params.device)
        return output_w

    def xfit(self, train_loader, val_loader, checkpoint=False):
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
            loss = 9999
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

            for i in trange(self.hidden_size):
                if i == self.hidden:
                    # success = True
                    Lambda = self.Lambdas[0]
                    new_hidden_weight = torch.empty(self.input_dim, 1).uniform_(-Lambda, Lambda).float().to(self.params.device)
                    new_bias_weight = torch.empty(1, 1).uniform_(-Lambda, Lambda).float().to(self.params.device)

                    new_fm = torch.mm(train_x, new_hidden_weight) + new_bias_weight
                    new_fm = torch.sigmoid(new_fm)

                    new_output_weight = self.solve_output(new_fm, fit_error)

                    if loss < self.tolerance:
                        break
                    else:
                        self.weight_IH = torch.cat((self.weight_IH, new_hidden_weight),1)
                        self.bias_IH = torch.cat((self.bias_IH, new_bias_weight), 1)
                        self.weight_HO = torch.cat((self.weight_HO, new_output_weight), 0)
                        self.hidden+=1
                
                if self.hidden == 1:
                    fm = torch.mm(train_x, self.weight_IH) + self.bias_IH
                    fm = torch.sigmoid(fm)
                    self.weight_HO = self.solve_output(fm, fit_error)

                pred = self.forward(train_x, i+1)
                fit_error = train_y - pred
                loss = self.loss_fn(pred, train_y).item()
                loss = np.sqrt(loss)
                self.loss_list.append(loss)

                vpred = self.forward(val_x, i+1)
                vloss = self.loss_fn(vpred, val_y).item()
                vloss = np.sqrt(vloss)
                self.vloss_list.append(vloss)


                self.logger.info('Hidden size: {} \t \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                    self.hidden, loss, vloss, min_vmse))

                if vloss < min_vmse:
                    min_vmse = vloss
                    self.best_hidden_size = self.hidden
                    self.best_HO = self.weight_HO.detach().clone()
                    self.logger.info('Found new best state')
                    self.logger.info('Best vmse: {:.4f}'.format(min_vmse))

                if checkpoint:
                    save_checkpoint({
                        'hidden_size': self.weight_IH.data.size(1),
                        'best_hidden_size': self.best_hidden_size,
                        'best_HO': self.best_HO,
                        'weight_IH': self.weight_IH,
                        'bias_IH': self.bias_IH,
                        'weight_HO': self.weight_HO,
                        'loss_list': self.loss_list,
                        'vloss_list': self.vloss_list}, checkpoint=self.params.task_dir, cv=self.params.cv)
                    self.logger.info(
                        'Checkpoint saved to {}'.format(self.params.task_dir))
                else:
                    self.best_state = {
                        'hidden_size': self.weight_IH.data.size(1),
                        'best_hidden_size': self.best_hidden_size,
                        'best_HO': self.best_HO,
                        'weight_IH': self.weight_IH,
                        'bias_IH': self.bias_IH,
                        'weight_HO': self.weight_HO,
                        'loss_list': self.loss_list,
                        'vloss_list': self.vloss_list}
                gc.collect()

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

        with torch.no_grad():
            input = torch.tensor(input).float().to(self.params.device)
            if using_best:
                pred= self.forward(input, self.best_hidden_size)
            else:
                pred = self.forward(input, self.hidden_size)
        return pred.cpu().numpy()
