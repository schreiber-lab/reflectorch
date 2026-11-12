import torch
from torch import Tensor

from reflectorch.data_generation.priors.no_constraints import DEFAULT_DEVICE, DEFAULT_DTYPE

__all__ = [
    "TimeGenerator",
    "UniformTimeGenerator",
    "RescaledTimeGenerator",
]


class TimeGenerator(object):
    """Base class for time generators"""
    def get_batch(self, batch_size: int) -> Tensor:
        pass

class UniformTimeGenerator(TimeGenerator):
    """Uniform time generator"""
    def __init__(self,
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 ):
        self.device = device
        self.dtype = dtype
    
    def get_batch(self, batch_size: int) -> Tensor:
        t = torch.rand((batch_size, 1), device=self.device, dtype=self.dtype)
        return t
    
class RescaledTimeGenerator(TimeGenerator):
    """Rescaled time generator"""
    def __init__(self,
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 alpha: float =1.0,
                 ):
        self.device = device
        self.dtype = dtype
        self.alpha = alpha
    
    def get_batch(self, batch_size: int) -> Tensor:
        t = torch.rand((batch_size, 1), device=self.device, dtype=self.dtype)
        t = t ** self.alpha
        return t