# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from reflectorch.data_generation.reflectivity.abeles import abeles_compiled, abeles
from reflectorch.data_generation.reflectivity.memory_eff import abeles_memory_eff
from reflectorch.data_generation.reflectivity.numpy_implementations import (
    kinematical_approximation_np,
    abeles_np,
)
from reflectorch.data_generation.reflectivity.smearing import abeles_constant_smearing
from reflectorch.data_generation.reflectivity.kinematical import kinematical_approximation


def reflectivity(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor = None,
        gauss_num: int = 51,
        constant_dq: bool = True,
        log: bool = False,
        abeles_func=None,
):
    """Function which computes the reflectivity from thin film parameters using the Abeles matrix formalism

    Args:
        q (Tensor): the momentum transfer (q) values
        thickness (Tensor): the layer thicknesses
        roughness (Tensor): the interlayer roughnesses
        sld (Tensor): the SLDs of the layers
        dq (Tensor, optional): the resolution for curve smearing. Defaults to None.
        gauss_num (int, optional): the number of gaussians for curve smearing. Defaults to 51.
        constant_dq (bool, optional): whether the smearing is constant. Defaults to True.
        log (bool, optional): if True the base 10 logarithm of the reflectivity curves is returned. Defaults to False.
        abeles_func (Callable, optional): a function implementing the simulation of the reflectivity curves, if different than the default implementation. Defaults to None.

    Returns:
        Tensor: the computed reflectivity curves
    """
    abeles_func = abeles_func or abeles
    q = torch.atleast_2d(q)

    if dq is None:
        reflectivity_curves = abeles_func(q, thickness, roughness, sld)
    else:
        reflectivity_curves = abeles_constant_smearing(
            q, thickness, roughness, sld,
            dq=dq, gauss_num=gauss_num, constant_dq=constant_dq, abeles_func=abeles_func
        )

    if log:
        reflectivity_curves = torch.log10(reflectivity_curves)
    return reflectivity_curves
