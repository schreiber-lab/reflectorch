from typing import List, Tuple

import torch
from torch import Tensor

from reflectorch.data_generation.utils import get_d_rhos, get_param_labels
from reflectorch.data_generation.reflectivity import reflectivity

__all__ = [
    "Params",
    "AbstractParams",
]


class AbstractParams(object):
    """Base class for parameters"""
    PARAM_NAMES: Tuple[str, ...]

    @staticmethod
    def rearrange_context_from_params(
            scaled_params: Tensor, context: Tensor, inference: bool = False, from_params: bool = False
    ):
        if inference:
            return context
        return scaled_params, context

    @staticmethod
    def restore_params_from_context(scaled_params: Tensor, context: Tensor):
        return scaled_params

    def reflectivity(self, q: Tensor, log: bool = False, **kwargs):
        """computes the reflectivity curves directly from the parameters"""
        raise NotImplementedError

    def __iter__(self):
        for name in self.PARAM_NAMES:
            yield getattr(self, name)

    def to_(self, tgt):
        for name, arr in zip(self.PARAM_NAMES, self):
            setattr(self, name, _to(arr, tgt))

    def to(self, tgt):
        """performs Pytorch Tensor dtype and/or device conversion"""
        return self.__class__(*[_to(arr, tgt) for arr in self])

    def cuda(self):
        """moves the parameters to the GPU"""
        return self.to('cuda')

    def cpu(self):
        """moves the parameters to the CPU"""
        return self.to('cpu')

    def __getitem__(self, item) -> 'AbstractParams':
        return self.__class__(*[
            arr.__getitem__(item) if isinstance(arr, Tensor) else arr for arr in self
        ])

    def __setitem__(self, key, other):
        if not isinstance(other, AbstractParams):
            raise ValueError

        for param, other_param in zip(self, other):
            if isinstance(param, Tensor):
                param[key] = other_param

    def __add__(self, other):
        if not isinstance(other, AbstractParams):
            raise NotImplemented

        return self.__class__(*[
            torch.cat([param, other_param], 0)
            if isinstance(param, Tensor) else param
            for param, other_param in zip(self, other)
        ])

    def __eq__(self, other):
        if not isinstance(other, AbstractParams):
            raise NotImplemented

        return all([torch.allclose(param, other_param) for param, other_param in zip(self, other)])

    @classmethod
    def cat(cls, *params: 'AbstractParams'):
        return cls(
            *[torch.cat([getattr(p, name) for p in params], 0)
              if isinstance(getattr(params[0], name), Tensor) else
              getattr(params[0], name)
              for name in cls.PARAM_NAMES]
        )

    @property
    def _ref_tensor(self):
        return getattr(self, self.PARAM_NAMES[0])

    def scale_with_q(self, q_ratio: float):
        raise NotImplementedError

    @property
    def max_layer_num(self) -> int:
        """gets the number of layers"""
        return self._ref_tensor.shape[-1]

    @property
    def batch_size(self) -> int:
        """gets the batch size"""
        return self._ref_tensor.shape[0]

    @property
    def device(self):
        """gets the Pytorch device"""
        return self._ref_tensor.device

    @property
    def dtype(self):
        """gets the Pytorch data type"""
        return self._ref_tensor.dtype

    @property
    def d_rhos(self):
        raise NotImplementedError

    def as_tensor(self, use_drho: bool = False) -> Tensor:
        """converts the instance of the class to a Pytorch tensor"""
        raise NotImplementedError

    @classmethod
    def from_tensor(cls, params: Tensor):
        """initializes an instance of the class from a Pytorch tensor"""
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        """get the number of parameters"""
        return self.layers_num2size(self.max_layer_num)

    @staticmethod
    def size2layers_num(size: int) -> int:
        """converts the number of parameters to the number of layers"""
        raise NotImplementedError

    @staticmethod
    def layers_num2size(layers_num: int) -> int:
        """converts the number of layers to the number of parameters"""
        raise NotImplementedError

    def get_param_labels(self) -> List[str]:
        """gets the parameter labels"""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'batch_size={self.batch_size}, ' \
               f'max_layer_num={self.max_layer_num}, ' \
               f'device={str(self.device)})'


def _to(arr, dest):
    if hasattr(arr, 'to'):
        arr = arr.to(dest)
    return arr


class Params(AbstractParams):
    """Parameter class for thickness, roughness and sld parameters

        Args:
            thicknesses (Tensor): batch of thicknesses (top to bottom)
            roughnesses (Tensor): batch of roughnesses (top to bottom)
            slds (Tensor): batch of slds (top to bottom)
        """
    MIN_THICKNESS: float = 0.5

    __slots__ = ('thicknesses', 'roughnesses', 'slds')
    PARAM_NAMES = __slots__

    def __init__(self, thicknesses: Tensor, roughnesses: Tensor, slds: Tensor):

        self.thicknesses = thicknesses
        self.roughnesses = roughnesses
        self.slds = slds

    @staticmethod
    def rearrange_context_from_params(
            scaled_params: Tensor, context: Tensor, inference: bool = False, from_params: bool = False
    ):
        if inference:
            return context
        return scaled_params, context

    @staticmethod
    def restore_params_from_context(scaled_params: Tensor, context: Tensor):
        return scaled_params

    def reflectivity(self, q: Tensor, log: bool = False, **kwargs):
        """computes the reflectivity curves directly from the parameters

        Args:
            q (Tensor): the q values
            log (bool, optional): whether to apply logarithm to the curves. Defaults to False.

        Returns:
            Tensor: the simulated reflectivity curves
        """
        return reflectivity(q, self.thicknesses, self.roughnesses, self.slds, log=log, **kwargs)

    def scale_with_q(self, q_ratio: float):
        """scales the parameters based on the q ratio

        Args:
            q_ratio (float): the scaling ratio
        """
        self.thicknesses /= q_ratio
        self.roughnesses /= q_ratio
        self.slds *= q_ratio ** 2

    @property
    def d_rhos(self):
        """computes the differences in SLD values of the neighboring layers"""
        return get_d_rhos(self.slds)

    def as_tensor(self, use_drho: bool = False) -> Tensor:
        """converts the instance of the class to a Pytorch tensor"""
        if use_drho:
            return torch.cat([self.thicknesses, self.roughnesses, self.d_rhos], -1)
        else:
            return torch.cat([self.thicknesses, self.roughnesses, self.slds], -1)

    @classmethod
    def from_tensor(cls, params: Tensor):
        """initializes an instance of the class from a Pytorch tensor containing the values of the parameters"""
        layers_num = (params.shape[-1] - 2) // 3

        thicknesses, roughnesses, slds = torch.split(params, [layers_num, layers_num + 1, layers_num + 1], dim=-1)

        return cls(thicknesses, roughnesses, slds)

    @staticmethod
    def size2layers_num(size: int) -> int:
        """converts the number of parameters to the number of layers"""
        return (size - 2) // 3

    @staticmethod
    def layers_num2size(layers_num: int) -> int:
        """converts the number of layers to the number of parameters"""
        return layers_num * 3 + 2

    def get_param_labels(self, **kwargs) -> List[str]:
        """gets the parameter labels, the layers are numbered from the bottom to the top 
           (i.e. opposite to the order in the Tensors)"""
        return get_param_labels(self.max_layer_num, **kwargs)
