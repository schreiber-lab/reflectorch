# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numpy import ndarray

from torch import Tensor, tensor

__all__ = [
    'to_np',
    'to_t',
    'angle_to_q',
    'q_to_angle',
    'energy_to_wavelength',
    'wavelength_to_energy',
]


def to_np(arr):
    if isinstance(arr, Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def to_t(arr, device=None, dtype=None):
    if not isinstance(arr, Tensor):
        return tensor(arr, device=device, dtype=dtype)
    return arr


# taken from mlreflect package
# mlreflect/xrrloader/dataloader/transform.py


def angle_to_q(scattering_angle: ndarray or float, wavelength: float):
    """Conversion from full scattering angle (degrees) to scattering vector (inverse angstroms)"""
    return 4 * np.pi / wavelength * np.sin(scattering_angle / 2 * np.pi / 180)


def q_to_angle(q: ndarray or float, wavelength: float):
    """Conversion from scattering vector (inverse angstroms) to full scattering angle (degrees)"""
    return 2 * np.arcsin(q * wavelength / (4 * np.pi)) / np.pi * 180


def energy_to_wavelength(energy: float):
    """Conversion from photon energy (eV) to photon wavelength (angstroms)"""
    return 1.2398 / energy * 1e4


def wavelength_to_energy(wavelength: float):
    """Conversion from photon wavelength (angstroms) to photon energy (eV)"""
    return 1.2398 / wavelength * 1e4
