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
    """Function which computes the reflectivity curves from thin film parameters. 
    By default it uses the fast implementation of the Abeles matrix formalism.

    Args:
        q (Tensor): tensor of momentum transfer (q) values with shape [batch_size, n_points] or [n_points]
        thickness (Tensor): tensor containing the layer thicknesses (ordered from top to bottom) with shape [batch_size, n_layers]
        roughness (Tensor): tensor containing the interlayer roughnesses (ordered from top to bottom) with shape [batch_size, n_layers + 1]
        sld (Tensor): tensors containing the layer SLDs (real or complex; ordered from top to bottom) with shape [batch_size, n_layers + 1]. 
                    It includes the substrate but excludes the ambient medium which is assumed to have an SLD of 0.
        dq (Tensor, optional): tensor of resolutions used for curve smearing with shape [batch_size, 1].
                            Either dq if ``constant_dq`` is ``True`` or dq/q if ``constant_dq`` is ``False``. Defaults to None.
        gauss_num (int, optional): the number of gaussians for curve smearing. Defaults to 51.
        constant_dq (bool, optional): if ``True`` the smearing is constant (constant dq at each point in the curve) 
                                    otherwise the smearing is linear (constant dq/q at each point in the curve). Defaults to True.
        log (bool, optional): if True the base 10 logarithm of the reflectivity curves is returned. Defaults to False.
        abeles_func (Callable, optional): a function implementing the simulation of the reflectivity curves, if different than the default Abeles matrix implementation. Defaults to None.

    Returns:
        Tensor: tensor containing the simulated reflectivity curves with shape [batch_size, n_points]
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
