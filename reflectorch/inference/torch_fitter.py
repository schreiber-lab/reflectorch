from tqdm import trange

import torch
from torch import nn, Tensor

from reflectorch.data_generation import LogLikelihood, reflectivity, PriorSampler


class ReflGradientFit(object):
    def __init__(self,
                 q: Tensor,
                 exp_curve: Tensor,
                 prior_sampler: PriorSampler,
                 params: Tensor,
                 fit_indices: Tensor,
                 sigmas: Tensor = None,
                 optim_cls=None,
                 lr: float = 1e-2,
                 rel_err: float = 0.1,
                 abs_err: float = 1e-7,
                 ):

        self.q = q

        if sigmas is None:
            sigmas = exp_curve * rel_err + abs_err

        self.likelihood = LogLikelihood(q, exp_curve, prior_sampler, sigmas)

        self.num_layers = params.shape[-1] // 3
        self.fit_indices = fit_indices
        self.init_params = params.clone()
        self.params_to_fit = nn.Parameter(self.init_params[fit_indices].clone())

        optim_cls = optim_cls or torch.optim.Adam
        self.optim = optim_cls([self.params_to_fit], lr)

        self.losses = []

    @property
    def params(self):
        params = self.init_params.clone()
        params[self.fit_indices] = self.params_to_fit
        return params

    def calc_log_likelihood(self):
        return self.likelihood.calc_log_likelihood(self.refl())

    def calc_log_prob_loss(self):
        return - self.calc_log_likelihood().mean()

    def refl(self):
        d, sigma, rho = torch.split(self.params, [self.num_layers, self.num_layers + 1, self.num_layers + 1], -1)
        return reflectivity(self.q, d, sigma, rho)

    def run(self, num_iterations: int = 500, disable_tqdm: bool = False):
        pbar = trange(num_iterations, disable=disable_tqdm)

        for _ in pbar:
            self.optim.zero_grad()
            loss = self.calc_log_prob_loss()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.item())
            pbar.set_description(f'Loss = {loss.item():.2e}')

    def clear(self):
        self.losses.clear()
