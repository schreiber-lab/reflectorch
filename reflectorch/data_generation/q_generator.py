from typing import Tuple, Union

import numpy as np

import torch
from torch import Tensor

from reflectorch.data_generation.utils import uniform_sampler
from reflectorch.data_generation.priors import BasicParams
from reflectorch.utils import angle_to_q
from reflectorch.data_generation.priors.no_constraints import DEFAULT_DEVICE, DEFAULT_DTYPE

__all__ = [
    "QGenerator",
    "ConstantQ",
    "VariableQ",
    "EquidistantQ",
    "ConstantAngle",
]


class QGenerator(object):
    """Base class for momentum transfer (q) generators"""
    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        pass


class ConstantQ(QGenerator):
    """Q generator for reflectivity curves with fixed discretization

    Args:
        q (Union[Tensor, Tuple[float, float, int]], optional): tuple (q_min, q_max, num_q) defining the minimum q value, maximum q value and the number of q points. Defaults to (0., 0.2, 128).
        device (optional): the Pytorch device. Defaults to DEFAULT_DEVICE.
        dtype (optional): the Pytorch data type. Defaults to DEFAULT_DTYPE.
        remove_zero (bool, optional): do not include the upper end of the interval. Defaults to False.
        fixed_zero (bool, optional): do not include the lower end of the interval. Defaults to False.
    """

    def __init__(self,
                 q: Union[Tensor, Tuple[float, float, int]] = (0., 0.2, 128),
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 remove_zero: bool = False,
                 fixed_zero: bool = False,
                 ):
        if isinstance(q, (tuple, list)):
            q = torch.linspace(*q, device=device, dtype=dtype)
            if remove_zero:
                if fixed_zero:
                    q = q[1:]
                else:
                    q = q[:-1]
        self.q_min = q.min().item()
        self.q_max = q.max().item()
        self.q = q

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        """generate a batch of q values

        Args:
            batch_size (int): the batch size

        Returns:
            Tensor: generated batch of q values
        """
        return self.q.clone()[None].expand(batch_size, self.q.shape[0])
    
    def scale_q(self, q):
        """Scales the q values to the range [-1, 1].

        Args:
            q (Tensor): unscaled q values

        Returns:
            Tensor: scaled q values
        """
        scaled_q_01 = (q - self.q_min) / (self.q_max - self.q_min)
        return 2.0 * (scaled_q_01 - 0.5)
    

class VariableQ(QGenerator):
    """Q generator for reflectivity curves with variable discretization

    Args:
        q_min_range (list, optional): the range for sampling the minimum q value of the curves, q_min. Defaults to [0.01, 0.03].
        q_max_range (list, optional): the range for sampling the maximum q value of the curves, q_max. Defaults to [0.1, 0.5].
        n_q_range (list, optional): the range for the number of points in the curves (equidistantly sampled between q_min and q_max, 
                                    the number of points varies between batches but is constant within a batch). Defaults to [64, 256].
        device (optional): the Pytorch device. Defaults to DEFAULT_DEVICE.
        dtype (optional): the Pytorch data type. Defaults to DEFAULT_DTYPE.
    """

    def __init__(self,
                 q_min_range: Tuple[float, float] = (0.01, 0.03),
                 q_max_range: Tuple[float, float] = (0.1, 0.5),
                 n_q_range: Tuple[int, int] = (64, 256),
                 mode: str = 'equidistant',
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 ):
        self.q_min_range = q_min_range
        self.q_max_range = q_max_range
        self.n_q_range = n_q_range
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        """generate a batch of q values (the number of points varies between batches but is constant within a batch)

        Args:
            batch_size (int): the batch size

        Returns:
            Tensor: generated batch of q values
        """

        q_min = torch.rand(batch_size, device=self.device, dtype=self.dtype) * (self.q_min_range[1] - self.q_min_range[0]) + self.q_min_range[0]
        q_max = torch.rand(batch_size, device=self.device, dtype=self.dtype) * (self.q_max_range[1] - self.q_max_range[0]) + self.q_max_range[0]

        n_q = torch.randint(self.n_q_range[0], self.n_q_range[1] + 1, (1,), device=self.device).item()

        if self.mode == 'equidistant':
            q = torch.linspace(0, 1, n_q, device=self.device, dtype=self.dtype)
        elif self.mode == 'random':
            q = torch.rand(n_q, device=self.device, dtype=self.dtype).sort().values

        q = q_min[:, None] + q * (q_max - q_min)[:, None]
        
        return q
    
    def scale_q(self, q):
        """scales the q values to the range [-1, 1]

        Args:
            q (Tensor): unscaled q values

        Returns:
            Tensor: scaled q values
        """
        scaled_q_01 = (q - self.q_min_range[0]) / (self.q_max_range[1] - self.q_min_range[0]) 

        return 2.0 * (scaled_q_01 - 0.5)


class ConstantAngle(QGenerator):
    """Q generator for reflectivity curves measured at equidistant angles

    Args:
        angle_range (Tuple[float, float, int], optional): the range of the incident angles. Defaults to (0., 0.2, 257).
        wavelength (float, optional): the beam wavelength in units of angstroms. Defaults to 1.
        device (optional): the Pytorch device. Defaults to DEFAULT_DEVICE.
        dtype (optional): the Pytorch data type. Defaults to DEFAULT_DTYPE.
        """
    def __init__(self,
                 angle_range: Tuple[float, float, int] = (0., 0.2, 257),
                 wavelength: float = 1.,
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 ):
        self.q = torch.from_numpy(angle_to_q(np.linspace(*angle_range), wavelength)).to(device).to(dtype)

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        """generate a batch of q values

        Args:
            batch_size (int): the batch size

        Returns:
            Tensor: generated batch of q values
        """
        return self.q.clone()[None].expand(batch_size, self.q.shape[0])


class EquidistantQ(QGenerator):
    def __init__(self,
                 max_range: Tuple[float, float],
                 num_values: Union[int, Tuple[int, int]],
                 device=None,
                 dtype=torch.float64
                 ):
        self.max_range = max_range
        self._num_values = num_values
        self.device = device
        self.dtype = dtype

    @property
    def num_values(self) -> int:
        if isinstance(self._num_values, int):
            return self._num_values
        return np.random.randint(*self._num_values)

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        num_values = self.num_values
        q_max = uniform_sampler(*self.max_range, batch_size, 1, device=self.device, dtype=self.dtype)
        norm_qs = torch.linspace(0, 1, num_values + 1, device=self.device, dtype=self.dtype)[1:][None]
        qs = norm_qs * q_max
        return qs


class TransformerQ(QGenerator):
    def __init__(self,
                 q_max: float = 0.2,
                 num_values: Union[int, Tuple[int, int]] = (30, 512),
                 min_dq_ratio: float = 5.,
                 device=None,
                 dtype=torch.float64,
                 ):
        self.min_dq_ratio = min_dq_ratio
        self.q_max = q_max
        self._dq_range = q_max / num_values[1], q_max / num_values[0]
        self._num_values = num_values
        self.device = device
        self.dtype = dtype

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        assert context is not None

        params: BasicParams = context['params']
        total_thickness = params.thicknesses.sum(-1)

        assert total_thickness.shape[0] == batch_size

        min_dqs = torch.clamp(
            2 * np.pi / total_thickness / self.min_dq_ratio, self._dq_range[0], self._dq_range[1] * 0.9
        )

        dqs = torch.rand_like(min_dqs) * (self._dq_range[1] - min_dqs) + min_dqs

        num_q_values = torch.clamp(self.q_max // dqs, *self._num_values).to(torch.int)

        q_values, mask = generate_q_padding_mask(num_q_values, self.q_max)

        context['tgt_key_padding_mask'] = mask
        context['num_q_values'] = num_q_values

        return q_values


def generate_q_padding_mask(num_q_values: Tensor, q_max: float):
    batch_size = num_q_values.shape[0]
    dqs = (q_max / num_q_values)[:, None]
    q_values = torch.arange(1, num_q_values.max().item() + 1)[None].repeat(batch_size, 1) * dqs
    mask = (q_values > q_max + dqs / 2)
    q_values[mask] = 0.
    return q_values, mask
