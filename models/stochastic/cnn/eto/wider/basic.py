import torch
import torch.nn as nn
from models.stochastic.Base import fcLayer, esnLayer, cnnLayer, cesLayer, escLayer
import gc

Arch_dict = {
    'esn': esnLayer,
    'cnn': cnnLayer,
    'ces': cesLayer,
    'esc': escLayer,
}


class cell(nn.Module):
    '''
    cell: opt class\n
        \tself.arch = arc_obj.best_arch\n
        \tself.ho = arc_obj.ho\n
        \tself.fc_io  = arc_obj.fc_io\n
    '''

    def __init__(self, arch, fc_io, ho=None):
        super(cell, self).__init__()

        self.arch = arch
        self.fc_io = fc_io
        self.ho = ho

    def hidden_forward(self, input):
        fm, fm_flatten = self.arch(input)
        fm_flatten = io_check(self.fc_io, fm_flatten, input)
        return fm, fm_flatten

    def forward(self, input):
        fm, fm_flatten = self.hidden_forward(input)
        pred = self.ho(fm_flatten)
        # fm = self.batchNorm(fm)
        return fm, fm_flatten, pred


class wide_cell(nn.Module):
    '''
    cell: opt class\n
        \tself.arch = arc_obj.best_arch\n
        \tself.ho = arc_obj.ho\n
        \tself.fc_io  = arc_obj.fc_io\n
    '''

    def __init__(self, arch, fc_io, ho=None):
        super().__init__()

        self.arch = arch
        self.fc_io = fc_io
        self.ho = ho

    def hidden_forward(self, input):
        _, fm_flatten = self.arch(input)
        fm_flatten = io_check(self.fc_io, fm_flatten, input)
        return fm_flatten

    def forward(self, input):
        pred = self.hidden_forward(input)
        pred = self.ho(pred)
        # fm = self.batchNorm(fm)
        return pred

    # @property
    # def dict(self):
    #     '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
    #     return self.__dict__


def cal_eNorm(_cell, dataPack):

    last_eNorm = torch.linalg.matrix_norm(dataPack.catE).item()

    if _cell.ho is None:
        _cell.ho = cell_ho(_cell, dataPack)

    # currentP = []
    # for _x in dataPack.x:
    #     _, _, pred = _cell(_x)
    #     currentP.append(pred)
    # catCurP = torch.cat(currentP, dim=0)

    catCurP = _cell(torch.cat(dataPack.x, dim=0))
    # catSumP = dataPack.catSumP + catCurP
    # catE = dataPack.catY - catSumP

    catE = dataPack.catE - catCurP

    curr_eNorm = torch.linalg.matrix_norm(catE).item()
    
    return last_eNorm, curr_eNorm


def close_ho(device, Hidden_States, y, lambda_reg=0., report=False):
    
    map_size = Hidden_States.size(1)
    H = y.size(1)
    
    t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
        device), Hidden_States), dim=1)
    HTH = t_hs.T @ t_hs
    HTY = t_hs.T @ y
    # ridge regression with pytorch
    I = (lambda_reg * torch.eye(HTH.size(0))).to(device)
    A = HTH + I
    # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
    tag = 'Inverse'
    try:
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        if orig_rank != t_hs.size(1):
            tag = 'Pseudo-inverse'
    except:
        tag = 'Pseudo-inverse'
    
    try:
        W = torch.linalg.lstsq(A.cpu(), HTY.cpu(), driver = 'gelsd').solution.T.to(device) # Attention this method may encounter bugs when above exception happens
    except:
        # If the computation of least square method meets bug, then make this subnetwork output zeros, tranferring the current error-feedback into the next construction iteration.
        W = torch.zeros(H, map_size+1).to(device)
        # When the training goes on, sometimes the input A may be singular, which makes the bug. This bug may be solved in pytorch 1.10. 

        # W = torch.mm(torch.linalg.pinv(A.cpu(), hermitian = True),
        #              HTY.cpu()).t().to(device)

    bias = nn.Parameter(W[:, 0], requires_grad=False)
    weight = nn.Parameter(W[:, 1:], requires_grad=False)
    if report:
        print('Global LSM: {} \t L2 regular: {}'.format(
            tag, 'True' if lambda_reg != 0 else 'False'))
    
    ho = fcLayer(map_size, H, device)
    ho.update(weight, bias)
    ho.freeze()
    ho.inverse = tag
    ho.lambda_reg = lambda_reg
    
    return ho

    

def cell_ho(_cell, data_pack, force_update=False, lambda_reg=0, report=False):
    if _cell.ho is not None and force_update is False:
        return _cell.ho

    cat_h = []
    for _x in data_pack.x:
        h = _cell.hidden_forward(_x)
        cat_h.append(h)

    cat_h = torch.cat(cat_h, dim=0)

    # cat_h = io_check(_cell.fc_io, cat_h, data_pack.catX)
    try:
        ho = close_ho(data_pack.device, cat_h,
                            data_pack.catE, lambda_reg=lambda_reg, report=report)
    except:
        ho = close_ho(data_pack.device, cat_h,
                            data_pack.catE, lambda_reg=1, report=report)

    torch.cuda.empty_cache() if next(ho.parameters()).is_cuda else gc.collect()

    return ho


def io_check(fc_io, hidden, x):
    if fc_io:
        Hidden = torch.cat((torch.flatten(x, start_dim=1), hidden), dim=1)
        return Hidden
    else:
        return hidden


def cnn_map(_hyper, steps):
    pooling_redcution = 0
    if _hyper.pooling:
        pooling_redcution = _hyper.pooling_size - 1
    conv_redcution = 0
    if _hyper.padding != 'same':
        conv_redcution = _hyper.kernel_size - 1

    map_size = steps - conv_redcution - pooling_redcution
    return map_size


def rnn_map(_hyper):
    map_size = _hyper.hidden_size
    return map_size
