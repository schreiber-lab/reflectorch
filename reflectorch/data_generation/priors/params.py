# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import Tensor

from reflectorch.data_generation.utils import get_d_rhos
from reflectorch.data_generation.reflectivity import reflectivity

__all__ = [
    "Params",
]


class Params(object):
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
        return reflectivity(q, self.thicknesses, self.roughnesses, self.slds, log=log, **kwargs)

    def __iter__(self):
        for name in self.PARAM_NAMES:
            yield getattr(self, name)

    def to_(self, tgt):
        for name, arr in zip(self.PARAM_NAMES, self):
            setattr(self, name, arr.to(tgt))

    def to(self, tgt):
        return self.__class__(*[arr.to(tgt) for arr in self])

    def __getitem__(self, item) -> 'Params':
        return self.__class__(*[arr.__getitem__(item) for arr in self])

    def __setitem__(self, key, other):
        if not isinstance(other, Params):
            raise ValueError

        for param, other_param in zip(self, other):
            param[key] = other_param

    def __add__(self, other):
        if not isinstance(other, Params):
            raise NotImplemented

        return self.__class__(*[
            torch.cat([param, other_param], 0) for param, other_param in zip(self, other)
        ])

    def __eq__(self, other):
        if not isinstance(other, Params):
            raise NotImplemented

        return all([torch.allclose(param, other_param) for param, other_param in zip(self, other)])

    @classmethod
    def cat(cls, *params: 'Params'):
        return cls(
            *[torch.cat([getattr(p, name) for p in params], 0)
              for name in cls.PARAM_NAMES]
        )

    def scale_with_q(self, q_ratio: float):
        self.thicknesses /= q_ratio
        self.roughnesses /= q_ratio
        self.slds *= q_ratio ** 2

    @property
    def max_layer_num(self) -> int:
        return self.thicknesses.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.thicknesses.shape[0]

    @property
    def device(self):
        return self.thicknesses.device

    @property
    def dtype(self):
        return self.thicknesses.dtype

    @property
    def d_rhos(self):
        return get_d_rhos(self.slds)

    def as_tensor(self, use_drho: bool = False) -> Tensor:
        if use_drho:
            return torch.cat([self.thicknesses, self.roughnesses, self.d_rhos], -1)
        else:
            return torch.cat([self.thicknesses, self.roughnesses, self.slds], -1)

    @classmethod
    def from_tensor(cls, params: Tensor):
        layers_num = (params.shape[-1] - 2) // 3

        thicknesses, roughnesses, slds = torch.split(params, [layers_num, layers_num + 1, layers_num + 1], dim=-1)

        return cls(thicknesses, roughnesses, slds)

    @property
    def num_params(self) -> int:
        return self.layers_num2size(self.max_layer_num)

    @staticmethod
    def size2layers_num(size: int) -> int:
        return (size - 2) // 3

    @staticmethod
    def layers_num2size(layers_num: int) -> int:
        return layers_num * 3 + 2

    def get_param_labels(self) -> List[str]:
        return get_param_labels(self.max_layer_num)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'batch_size={self.batch_size}, ' \
               f'max_layer_num={self.max_layer_num}, ' \
               f'device={str(self.device)})'


def get_param_labels(num_layers: int) -> List[str]:
    sld_units = '($10^{{-6}}$ Å$^{{-2}}$)'
    thickness_labels = [fr'$d_{i + 1}$ (Å)' for i in range(num_layers)]
    roughness_labels = [fr'$\sigma_{i + 1}$ (Å)' for i in range(num_layers)] + [r'$\sigma_{sub}$ (Å)']
    sld_labels = [fr'$\rho_{i + 1}$ {sld_units}' for i in range(num_layers)] + [fr'$\rho_{{sub}}$ {sld_units}']
    return thickness_labels + roughness_labels + sld_labels
