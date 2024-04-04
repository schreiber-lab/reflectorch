from torch import nn

ACTIVATIONS = {
    'relu': nn.ReLU(),
    'lrelu': nn.LeakyReLU(),
    'gelu': nn.GELU(),
    'selu': nn.SELU(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    
    'silu': nn.SiLU(), #added
    'mish': nn.Mish(),
}


def activation_by_name(name):
    """returns an activation function module corresponding to its name

    Args:
        name (str): string denoting the activation function ('relu', 'lrelu', 'gelu', 'selu', 'elu', 'sigmoid', 'silu', 'mish')

    Returns:
        nn.Module: Pytorch activation function module
    """
    if not isinstance(name, str):
        return name
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError:
        raise KeyError(f'Unknown activation function {name}')
