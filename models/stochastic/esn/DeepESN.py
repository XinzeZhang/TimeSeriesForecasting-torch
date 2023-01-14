import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from re import S
import gc
from models.stochastic.Base import esnLayer
from models.stochastic.esn.ESN import EchoStateNetwork
import torch
import torch.nn as nn


class Deep_ESN(EchoStateNetwork):
    """
    Gallicchio, Claudio, and Alessio Micheli. "Echo state property of deep reservoir computing networks." Cognitive Computation 9.3 (2017): 337-350.
    """

    def __init__(self, opts, logger):
        self.num_layers = opts.num_layers
        super().__init__(opts=opts, logger=logger)

    def init_arch(self,):
        self.stack_layers = nn.ModuleList([])
        for layer in range(self.num_layers):
            layer_input_dim = self.Input_dim if layer == 0 else self.Hidden_Size
            layer_esn = esnLayer(
                hidden_size=self.Hidden_Size,
                input_dim=layer_input_dim,
                nonlinearity=self.opts.nonlinearity,
                leaky_r=self.opts.leaky_r,
                weight_scale=self.opts.weight_scaling,
                iw_bound=self.opts.iw_bound,
                hw_bound=self.opts.hw_bound,
                device=self.device
            )
            layer_esn.freeze()
            self.stack_layers.append(layer_esn)

        self.readout_size = self.num_layers * self.Hidden_Size
        if self.fc_io == 'step':
            self.readout_size += self.Input_dim
        elif self.fc_io == 'series':
            self.readout_size += self.Hidden_Size + self.Lag_order
            
        self.readout = nn.Linear(self.readout_size, self.Output_dim)
        self.readout.weight.requires_grad = False
        self.readout.bias.requires_grad = False

    def layer_transform(self, x, layer):
        '得到Layer层的hidden_state和最后一个state'
        layer_hidden_state, last_state = self.stack_layers[layer](x)
        return layer_hidden_state, last_state

    def reservior_transform(self, input_state):
        Hidden_States = []
        for layer in range(self.num_layers):
            input_state, _ = self.layer_transform(
                input_state, layer)
            Hidden_States.append(input_state)

            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        Hidden_States = torch.cat(Hidden_States, dim=1)

        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        return Hidden_States

    def xfit_logger(self,):
        self.logger.info('\nLayers: {} \t Hidden size: {} \t Readout size: {} \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
            self.num_layers, self.Hidden_Size, self.readout_size, self.fit_info.trmse, self.fit_info.vrmse))