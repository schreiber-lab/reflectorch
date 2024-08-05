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
    """Converts Pytorch tensor or Python list to Numpy array

    Args:
        arr (torch.Tensor or list): Input Pytorch tensor or Python list

    Returns:
        numpy.ndarray: Converted Numpy array
    """
    
    if isinstance(arr, Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def to_t(arr, device=None, dtype=None):
    """Converts Numpy array or Python list to Pytorch tensor

    Args:
        arr (numpy.ndarray or list): Input 
        device (torch.device or str, optional): device for the tensor ('cpu', 'cuda')
        dtype (torch.dtype, optional): data type of the tensor (e.g. torch.float32)

    Returns:
        torch.Tensor: converted Pytorch tensor
    """
    
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
