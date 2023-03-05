from tqdm import trange

from time import perf_counter

import torch
from torch import nn, Tensor

from reflectorch.data_generation.reflectivity import kinematical_approximation
from reflectorch.data_generation.priors.multilayer_structures import SimpleMultilayerSampler


class MultilayerFit(object):
    def __init__(self,
                 q: Tensor,
                 exp_curve: Tensor,
                 params: Tensor,
                 convert_func,
                 scale_curve_func,
                 min_bounds: Tensor,
                 max_bounds: Tensor,
                 optim_cls=None,
                 lr: float = 5e-2,
                 ):

        self.q = q
        self.scale_curve_func = scale_curve_func
        self.scaled_exp_curve = self.scale_curve_func(torch.atleast_2d(exp_curve))
        self.params_to_fit = nn.Parameter(params.clone())
        self.convert_func = convert_func

        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        optim_cls = optim_cls or torch.optim.Adam
        self.optim = optim_cls([self.params_to_fit], lr)

        self.best_loss = float('inf')
        self.best_params, self.best_curve = None, None

        self.get_best_solution()
        self.losses = []

    @classmethod
    def from_prior_sampler(
            cls,
            q: Tensor,
            exp_curve: Tensor,
            prior_sampler: SimpleMultilayerSampler,
            batch_size: int = 2 ** 13,
            **kwargs
    ):
        _, scaled_params = prior_sampler.optimized_sample(batch_size)
        params = prior_sampler.restore_params2parametrized(scaled_params)

        return cls(
            q.float(), exp_curve.float(), params.float(),
            convert_func=get_convert_func(prior_sampler.multilayer_model),
            scale_curve_func=_save_log, min_bounds=prior_sampler.min_bounds,
            max_bounds=prior_sampler.max_bounds,
            **kwargs
        )

    @classmethod
    def from_prediction(
            cls,
            param: Tensor,
            prior_sampler,
            q: Tensor,
            exp_curve: Tensor,
            batch_size: int = 2 ** 13,
            rel_bounds: float = 0.3,
            **kwargs
    ):

        num_params = param.shape[-1]

        deltas = prior_sampler.restore_params2parametrized(
            (torch.rand(batch_size, num_params, device=param.device) * 2 - 1) * rel_bounds
        ) - prior_sampler.min_bounds

        deltas[0] = 0.

        params = torch.atleast_2d(param) + deltas

        return cls(
            q.float(), exp_curve.float(), params.float(),
            convert_func=get_convert_func(prior_sampler.multilayer_model),
            scale_curve_func=_save_log, min_bounds=prior_sampler.min_bounds,
            max_bounds=prior_sampler.max_bounds,
            **kwargs
        )

    def get_clipped_params(self):
        return torch.clamp(self.params_to_fit.detach().clone(), self.min_bounds, self.max_bounds)

    def calc_loss(self, reduce=True, clip: bool = False):
        if clip:
            params = self.get_clipped_params()
        else:
            params = self.params_to_fit
        curves = self.get_curves(params=params)
        losses = ((self.scale_curve_func(curves) - self.scaled_exp_curve) ** 2)

        if reduce:
            return losses.sum()
        else:
            return losses.sum(-1)

    def get_curves(self, params: Tensor = None):
        if params is None:
            params = self.params_to_fit
        curves = kinematical_approximation(self.q, **self.convert_func(params))
        return torch.atleast_2d(curves)

    def run(self, num_iterations: int = 500, disable_tqdm: bool = False):
        pbar = trange(num_iterations, disable=disable_tqdm)

        for _ in pbar:
            self.optim.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.item())
            pbar.set_description(f'Loss = {loss.item():.2e}')

    def clear(self):
        self.losses.clear()

    def run_fixed_time(self, time_limit: float = 2.):

        start = perf_counter()

        while True:
            self.optim.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.item())
            time_spent = perf_counter() - start

            if time_spent > time_limit:
                break

    @torch.no_grad()
    def get_best_solution(self, clip: bool = True):
        losses = self.calc_loss(clip=clip, reduce=False)
        idx = torch.argmin(losses)
        best_loss = losses[idx].item()

        if best_loss < self.best_loss:
            best_curve = self.get_curves()[idx]
            best_params = self.params_to_fit.detach()[idx].clone()
            self.best_params, self.best_curve, self.best_loss = best_params, best_curve, best_loss

        return self.best_params


def get_convert_func(multilayer_model):
    def func(params):
        params = multilayer_model.to_standard_params(params)
        return {
            'thickness': params['thicknesses'],  # very useful transformation indeed ...
            'roughness': params['roughnesses'],
            'sld': params['slds'],
        }

    return func


def _save_log(curves, eps: float = 1e-10):
    return torch.log10(curves + eps)
