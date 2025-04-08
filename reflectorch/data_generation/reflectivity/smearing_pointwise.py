import torch
import scipy
import numpy as np
from functools import lru_cache
from typing import Tuple

from reflectorch.data_generation.reflectivity.abeles import abeles

#Pytorch version based on the JAX implementation of pointwise smearing in the refnx package.

@lru_cache(maxsize=128)
def gauss_legendre(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Gaussian quadrature abscissae and weights.

    Args:
        n (int): Gaussian quadrature order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The abscissae and weights for Gauss-Legendre integration.
    """
    return scipy.special.p_roots(n)

def gauss(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Gaussian function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the Gaussian function.
    """
    return torch.exp(-0.5 * x * x)

def abeles_pointwise_smearing(
    q: torch.Tensor,
    dq: torch.Tensor,
    thickness: torch.Tensor,
    roughness: torch.Tensor,
    sld: torch.Tensor,
    gauss_num: int = 17,
    abeles_func=None,
    **abeles_kwargs,
) -> torch.Tensor:
    """
    Compute reflectivity with variable smearing using Gaussian quadrature.

    Args:
        q (torch.Tensor): The momentum transfer (q) values.
        dq (torch.Tensor): The resolution for curve smearing.
        thickness (torch.Tensor): The layer thicknesses.
        roughness (torch.Tensor): The interlayer roughnesses.
        sld (torch.Tensor): The SLDs of the layers.
        sld_magnetic (torch.Tensor, optional): The magnetic SLDs of the layers.
        magnetization_angle (torch.Tensor, optional): The magnetization angles.
        polarizer_eff (torch.Tensor, optional): The polarizer efficiency.
        analyzer_eff (torch.Tensor, optional): The analyzer efficiency.
        abeles_func (Callable, optional): A function implementing the simulation of the reflectivity curves.
        gauss_num (int, optional): Gaussian quadrature order. Defaults to 17.

    Returns:
        torch.Tensor: The computed reflectivity curves.
    """
    abeles_func = abeles_func or abeles

    if q.shape[0] == 1:
        q = q.repeat(thickness.shape[0], 1)

    _FWHM = 2 * np.sqrt(2 * np.log(2.0))
    _INTLIMIT = 3.5

    bs = q.shape[0]
    nq = q.shape[-1]
    device = q.device

    quad_order = gauss_num
    abscissa, weights = gauss_legendre(quad_order)
    abscissa = torch.tensor(abscissa)[None, :, None].to(device)
    weights = torch.tensor(weights)[None, :, None].to(device)
    prefactor = 1.0 / np.sqrt(2 * np.pi)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    va = q[:, None, :] - _INTLIMIT * dq[:, None, :] / _FWHM
    vb = q[:, None, :] + _INTLIMIT * dq[:, None, :] / _FWHM

    qvals_for_res_0 = (abscissa * (vb - va) + vb + va) / 2
    qvals_for_res = qvals_for_res_0.reshape(bs, -1)

    refl_curves = abeles_func(
        q=qvals_for_res,
        thickness=thickness,
        roughness=roughness,
        sld=sld,
        **abeles_kwargs
    )

    # Handle multiple channels
    if refl_curves.dim() == 3:
        n_channels = refl_curves.shape[1]
        refl_curves = refl_curves.reshape(bs, n_channels, quad_order, nq)
        refl_curves = refl_curves * gaussvals.unsqueeze(1) * weights.unsqueeze(1)
        refl_curves = torch.sum(refl_curves, dim=2) * _INTLIMIT
    else:
        refl_curves = refl_curves.reshape(bs, quad_order, nq)
        refl_curves = refl_curves * gaussvals * weights
        refl_curves = torch.sum(refl_curves, dim=1) * _INTLIMIT

    return refl_curves