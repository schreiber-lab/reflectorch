# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import numpy as np

import torch
from torch import Tensor

from reflectorch.data_generation.utils import uniform_sampler
from reflectorch.data_generation.priors import Params
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
    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        pass


class ConstantQ(QGenerator):
    def __init__(self,
                 q: Union[Tensor, Tuple[float, float, int]] = (0., 0.2, 257),
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 remove_zero: bool = True,
                 fixed_zero: bool = False,  # bug from previous version left for back-compatibility
                 ):
        if isinstance(q, (tuple, list)):
            q = torch.linspace(*q, device=device, dtype=dtype)
            if remove_zero:
                if fixed_zero:
                    q = q[1:]
                else:
                    q = q[:-1]
        self.q = q

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        return self.q.clone()[None].expand(batch_size, self.q.shape[0])
    
    
# class VariableQ(QGenerator):
#     def __init__(self,
#                  q_min_range = [0.01, 0.03],
#                  q_max_range = [0.1, 0.5],
#                  n_q_range = [64, 512],
#                  device=DEFAULT_DEVICE,
#                  dtype=DEFAULT_DTYPE,
#                  ):
#         self.q_min_range = q_min_range
#         self.q_max_range = q_max_range
#         self.n_q_range = n_q_range
#         self.device = device
#         self.dtype = dtype

#     def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
#         q_min = np.random.uniform(*self.q_min_range)
#         q_max = np.random.uniform(*self.q_max_range)
#         n_q = np.random.randint(*self.n_q_range)
        
#         q = torch.linspace(start=q_min, end=q_max, steps=n_q).to(self.device).to(self.dtype)
        
#         return q[None].expand(batch_size, q.shape[0])

class VariableQ(QGenerator):
    def __init__(self,
                 q_min_range = [0.01, 0.03],
                 q_max_range = [0.1, 0.5],
                 n_q_range = [64, 512],
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 ):
        self.q_min_range = q_min_range
        self.q_max_range = q_max_range
        self.n_q_range = n_q_range
        self.device = device
        self.dtype = dtype

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
        q_min = np.random.uniform(*self.q_min_range, batch_size)
        q_max = np.random.uniform(*self.q_max_range, batch_size)
        n_q = np.random.randint(*self.n_q_range)
        
        q = torch.from_numpy(np.linspace(q_min, q_max, n_q).T).to(self.device).to(self.dtype)
        
        return q


class ConstantAngle(QGenerator):
    def __init__(self,
                 angle_range: Tuple[float, float, int] = (0., 0.2, 257),
                 wavelength: float = 1.,
                 device=DEFAULT_DEVICE,
                 dtype=DEFAULT_DTYPE,
                 ):
        self.q = torch.from_numpy(angle_to_q(np.linspace(*angle_range), wavelength)).to(device).to(dtype)

    def get_batch(self, batch_size: int, context: dict = None) -> Tensor:
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

        params: Params = context['params']
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
