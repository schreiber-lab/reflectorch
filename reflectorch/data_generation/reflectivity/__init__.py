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
from reflectorch.data_generation.reflectivity.smearing_pointwise import abeles_pointwise_smearing
from reflectorch.data_generation.reflectivity.kinematical import kinematical_approximation


def reflectivity(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor = None,
        gauss_num: int = 51,
        constant_dq: bool = False,
        log: bool = False,
        q_shift: Tensor = 0.0,
        r_scale: Tensor = 1.0,
        background: Tensor = 0.0,
        solvent_vf = None,
        solvent_mode = 'fronting',
        abeles_func = None,
        **abeles_kwargs
):
    """Function which computes the reflectivity curves from thin film parameters. 
    By default it uses the fast implementation of the Abeles matrix formalism.

    Args:
        q (Tensor): tensor of momentum transfer (q) values with shape [batch_size, n_points] or [n_points]
        thickness (Tensor): tensor containing the layer thicknesses (ordered from top to bottom) with shape [batch_size, n_layers]
        roughness (Tensor): tensor containing the interlayer roughnesses (ordered from top to bottom) with shape [batch_size, n_layers + 1]
        sld (Tensor): tensor containing the layer SLDs (real or complex; ordered from top to bottom) with shape 
            [batch_size, n_layers + 1] (excluding ambient SLD which is assumed to be 0)  or [batch_size, n_layers + 2] (including ambient SLD; only for the default ``abeles_func='abeles'``) 
        dq (Tensor, optional): tensor of resolutions used for curve smearing with shape [batch_size, 1].
                            Either dq if ``constant_dq`` is ``True`` or dq/q if ``constant_dq`` is ``False``. Defaults to None.
        gauss_num (int, optional): the number of gaussians for curve smearing. Defaults to 51.
        constant_dq (bool, optional): if ``True`` the smearing is constant (constant dq at each point in the curve) 
                                    otherwise the smearing is linear (constant dq/q at each point in the curve). Defaults to False.
        log (bool, optional): if True the base 10 logarithm of the reflectivity curves is returned. Defaults to False.
        q_shift (float or Tensor, optional): misalignment in q.
        r_scale (float or Tensor, optional): normalization factor (scales reflectivity).
        background (float or Tensor, optional): background intensity.
        abeles_func (Callable, optional): a function implementing the simulation of the reflectivity curves, if different than the default Abeles matrix implementation ('abeles'). Defaults to None.
        abeles_kwargs: Additional arguments specific to the chosen `abeles_func`.
    Returns:
        Tensor: the computed reflectivity curves
    """
    abeles_func = abeles_func or abeles
    q = torch.atleast_2d(q) + q_shift
    q = torch.clamp(q, min=0.0)
    
    if solvent_vf is not None:
        num_layers = thickness.shape[-1]
        if solvent_mode == 'fronting':
            assert sld.shape[-1] == num_layers + 2
            assert solvent_vf.shape[-1] == num_layers
            solvent_sld = sld[..., [0]]
            idx = slice(1, num_layers)
            sld[..., idx] = solvent_vf * solvent_sld + (1.0 - solvent_vf) * sld[..., idx]
        elif solvent_mode == 'backing':
            solvent_sld = sld[..., [-1]]
            idx = slice(1, num_layers) if sld.shape[-1] == num_layers + 2 else slice(0, num_layers)
            sld[..., idx] = solvent_vf * solvent_sld + (1.0 - solvent_vf) * sld[..., idx]
        else:
            raise NotImplementedError

    if dq is None:
        reflectivity_curves = abeles_func(q, thickness, roughness, sld, **abeles_kwargs)
    else:
        if dq.shape[-1] > 1:
            reflectivity_curves = abeles_pointwise_smearing(
                    q=q, dq=dq, thickness=thickness, roughness=roughness, sld=sld, 
                    abeles_func=abeles_func, gauss_num=gauss_num,
                    **abeles_kwargs,
            )
        else:
            reflectivity_curves = abeles_constant_smearing(
                q, thickness, roughness, sld,
                dq=dq, gauss_num=gauss_num, constant_dq=constant_dq, abeles_func=abeles_func,
                **abeles_kwargs,
            )

    if isinstance(r_scale, Tensor):
        r_scale = r_scale.view(-1, *[1] * (reflectivity_curves.dim() - 1))
    if isinstance(background, Tensor):
        background = background.view(-1, *[1] * (reflectivity_curves.dim() - 1))

    reflectivity_curves = reflectivity_curves * r_scale + background

    if log:
        reflectivity_curves = torch.log10(reflectivity_curves)

    return reflectivity_curves
