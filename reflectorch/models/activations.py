import torch
from torch import nn
from torch.nn.functional import relu

class Rowdy(nn.Module):
    """adaptive activation function"""
    def __init__(self, K=9):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.cat((torch.ones(1), torch.zeros(K-1))))  
        self.alpha.requiresGrad = True
        self.omega = nn.Parameter(torch.ones(K))  
        self.omega.requiresGrad = True

    def forward(self, x):
        rowdy = self.alpha[0]*relu(self.omega[0]*x)
        for k in range(1, self.K):
            rowdy += self.alpha[k]*torch.sin(self.omega[k]*k*x)
        return rowdy
    

ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,    
    'tanh': nn.Tanh,
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'rowdy': Rowdy,
}


def activation_by_name(name):
    """returns an activation function module corresponding to its name

    Args:
        name (str): string denoting the activation function ('relu', 'lrelu', 'gelu', 'selu', 'elu', 'sigmoid', 'silu', 'mish', 'rowdy')

    Returns:
        nn.Module: Pytorch activation function module
    """
    if not isinstance(name, str):
        return name
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError:
        raise KeyError(f'Unknown activation function {name}')
