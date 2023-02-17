# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from torch import Tensor, tensor

__all__ = [
    'to_np',
    'to_t',
]


def to_np(arr):
    if isinstance(arr, Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def to_t(arr, device=None, dtype=None):
    if not isinstance(arr, Tensor):
        return tensor(arr, device=device, dtype=dtype)
    return arr
