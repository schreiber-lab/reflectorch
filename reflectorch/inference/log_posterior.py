import torch
import numpy as np
from typing import Any, Optional, Union

class ReflectivityLogPosterior:
    """Callable log-posterior for reflectivity inference"""
    def __init__(
        self,
        param_model: Any,
        q_exp: Union[np.ndarray, torch.Tensor],
        curve_exp: Union[np.ndarray, torch.Tensor],
        prior_bounds: Union[np.ndarray, torch.Tensor],  # [dim, 2]
        device: Union[str, torch.device] = "cpu",
        sigmas_exp: Optional[Union[np.ndarray, torch.Tensor]] = None,
        q_resolution: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        rel_err_fallback: float = 0.3,
        ambient_sld: Optional[Union[float, torch.Tensor]] = None,
        sld_indices: Optional[Union[slice, np.ndarray, list]] = None,
        eps_inside: float = 1e-10,
    ):
        self.param_model = param_model
        self.device = torch.device(device)

        self.eps_inside = eps_inside
        self.rel_err_fallback = rel_err_fallback

        self.q_exp_t = torch.as_tensor(q_exp, dtype=torch.float64, device=self.device)
        self.curve_exp_t = torch.as_tensor(curve_exp, device=self.device)

        pb = torch.as_tensor(prior_bounds, device=self.device)
        self.min_bounds = pb[..., 0]
        self.max_bounds = pb[..., 1]

        FIXED_TOL = 1e-7

        widths = self.max_bounds - self.min_bounds
        fixed = widths.abs() <= FIXED_TOL
        self.fixed_mask = fixed
        self.free_mask = ~fixed

        safe_widths = widths.clone()
        safe_widths[fixed] = 1.0

        self.log_prior_const = -torch.sum(torch.log(safe_widths))

        if sigmas_exp is None:
            self.sigmas_t = self.curve_exp_t * self.rel_err_fallback + 1e-11
            self.exp_error_bars_available = False
        else:
            self.sigmas_t = torch.as_tensor(sigmas_exp, device=self.device)
            self.exp_error_bars_available = True

        if q_resolution is None:
            self.dq_t = None
        else:
            self.dq_t = torch.atleast_2d(torch.as_tensor(q_resolution, device=self.device))

        self.sld_indices = sld_indices
        if ambient_sld is None:
            self.ambient_sld_t = None
        else:
            self.ambient_sld_t = torch.atleast_2d(torch.as_tensor(ambient_sld, device=self.device))

        self._log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

    def set_rel_err_fallback(self, rel_err_fallback: float = 0.3) -> None:
        assert self.exp_error_bars_available is False, "Experimental error bars are available"

        self.rel_err_fallback = rel_err_fallback
        self.sigmas_t = self.curve_exp_t * self.rel_err_fallback + 1e-11

    @torch.inference_mode()
    def _apply_ambient_shift(self, theta: torch.Tensor) -> torch.Tensor:
        if self.ambient_sld_t is None or self.sld_indices is None:
            return theta
        theta_sim = theta.clone()
        theta_sim[..., self.sld_indices] = theta_sim[..., self.sld_indices] - self.ambient_sld_t
        return theta_sim

    @torch.inference_mode()
    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Uniform box prior, ignoring fixed parameters."""
        if theta.ndim == 1:
            theta = theta[None, :]

        if self.free_mask.any():
            inside = (
                (theta[..., self.free_mask] >= self.min_bounds[self.free_mask] - self.eps_inside) &
                (theta[..., self.free_mask] <= self.max_bounds[self.free_mask] + self.eps_inside)
            ).all(dim=-1)
        else:
            inside = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)

        lp = torch.where(
            inside,
            torch.full(
                (theta.shape[0],),
                self.log_prior_const,
                dtype=theta.dtype,
                device=theta.device,
            ),
            torch.full(
                (theta.shape[0],),
                -float("inf"),
                dtype=theta.dtype,
                device=theta.device,
            ),
        )

        return lp

    @torch.inference_mode()
    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Gaussian log-likelihood with fixed parameters clamped."""
        if theta.ndim == 1:
            theta = theta[None, :]

        theta_full = theta.clone()
        if self.fixed_mask.any():
            theta_full[..., self.fixed_mask] = self.min_bounds[self.fixed_mask]

        theta_sim = self._apply_ambient_shift(theta_full)

        dq_used = None
        if self.dq_t is not None:
            if self.dq_t.shape[0] == 1 and theta_full.shape[0] > 1:
                dq_used = self.dq_t.expand(theta_full.shape[0], *self.dq_t.shape[1:])
            else:
                dq_used = self.dq_t

        refl_kwargs = {}
        if dq_used is not None:
            refl_kwargs["dq"] = dq_used

        curves_sim = self.param_model.reflectivity(
            q=self.q_exp_t,
            parametrized_model=theta_sim,
            **refl_kwargs,
        )

        s = self.sigmas_t
        resid = (curves_sim - self.curve_exp_t) / s

        sum_dims = tuple(range(1, resid.ndim))
        ll = -0.5 * torch.sum(
            resid**2 + 2.0 * torch.log(s) + self._log2pi,
            dim=sum_dims,
        )

        return ll

    @torch.inference_mode()
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """log-posterior = log-likelihood + log-prior"""
        return self.log_likelihood(theta) + self.log_prior(theta)