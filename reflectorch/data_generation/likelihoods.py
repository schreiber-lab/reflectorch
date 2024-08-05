from typing import Union, Tuple

import torch
from torch import Tensor

from reflectorch.data_generation import (
    PriorSampler,
    Params,
)


class LogLikelihood(object):
    """Computes the gaussian log likelihood of the thin film parameters

    Args:
        q (Tensor): the q values
        exp_curve (Tensor): the experimental reflectivity curve
        priors (PriorSampler): the prior sampler
        sigmas (Union[float, Tensor]): the sigmas (i.e. intensity error bars)
    """
    def __init__(self, q: Tensor, exp_curve: Tensor, priors: PriorSampler, sigmas: Union[float, Tensor]):
        self.exp_curve = torch.atleast_2d(exp_curve)
        self.priors: PriorSampler = priors
        self.q = q
        self.sigmas = sigmas
        self.sigmas2 = self.sigmas ** 2

    def calc_log_likelihood(self, curves: Tensor):
        "computes the gaussian log likelihood"
        log_probs = - (self.exp_curve - curves) ** 2 / self.sigmas2 / 2
        return log_probs.sum(-1)

    def __call__(self, params: Union[Params, Tensor], curves: Tensor = None):
        if not isinstance(params, Params):
            params: Params = self.priors.PARAM_CLS.from_tensor(params)
        log_priors: Tensor = self.priors.log_prob(params)
        indices: Tensor = torch.isfinite(log_priors)

        if not indices.sum().item():
            return log_priors

        finite_params: Params = params[indices]

        if curves is None:
            curves: Tensor = finite_params.reflectivity(self.q)
        else:
            curves = curves[indices]

        log_priors[indices] += self.calc_log_likelihood(curves)

        return log_priors

    calc_log_posterior = __call__

    def get_importance_sampling_weights(
            self, sampled_params: Params, nf_log_probs: Tensor, curves: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        log_probs = self.calc_log_posterior(sampled_params, curves=curves)
        log_weights = log_probs - nf_log_probs
        log_weights = log_weights - log_weights.max()

        weights = torch.exp(log_weights.to(torch.float64)).to(log_weights)
        weights = weights / weights.sum()

        return weights, log_weights, log_probs


class PoissonLogLikelihood(LogLikelihood):
    """Computes the Poisson log likelihood of the thin film parameters

    Args:
        q (Tensor): the q values
        exp_curve (Tensor): the experimental reflectivity curve
        priors (PriorSampler): the prior sampler
        sigmas (Union[float, Tensor]): the sigmas (i.e. intensity error bars)
    """    
    def calc_log_likelihood(self, curves: Tensor):
        """computes the Poisson log likelihood"""
        log_probs = self.exp_curve / self.sigmas2 * (self.exp_curve * torch.log(curves) - curves)
        return log_probs.sum(-1)
