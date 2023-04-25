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
