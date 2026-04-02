import warnings
import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize, curve_fit
import torch

from reflectorch.data_generation.priors.base import PriorSampler

__all__ = [
    "refl_fit",
    "get_fit_with_growth",
]

def refl_fit(
    q: np.ndarray,
    curve: np.ndarray,
    init_params: np.ndarray,
    prior_sampler: PriorSampler,
    bounds: np.ndarray = None,
    error_bars: np.ndarray = None,
    scale_curve_func=np.log10,
    method: str = "trf",
    polishing_max_nfev: int = None,
    reflectivity_kwargs: dict = None,
    curve_func=None,
    return_param_errors: bool = False,
    **kwargs
):
    """
    Fit a reflectivity curve using a generic parametric model exposed through
    ``prior_sampler.param_model``.

    Args:
        q:
            Experimental q-values of shape ``(n_q,)``.
        curve:
            Experimental reflectivity curve of shape ``(n_q,)``.
        init_params:
            Initial parameter vector.
        prior_sampler:
            Prior sampler whose ``param_model`` provides the reflectivity model.
        bounds:
            Optional bounds array of shape ``(2, n_params)`` as expected by
            ``scipy.optimize.curve_fit``.
        error_bars:
            Optional uncertainties on the reflectivity values.
        scale_curve_func:
            Transformation applied before fitting, e.g. ``np.log10``.
        method:
            Optimization method for ``curve_fit``.
        polishing_max_nfev:
            Maximum number of function evaluations.
        reflectivity_kwargs:
            Optional extra keyword arguments passed to the reflectivity model.
        curve_func:
            Optional custom forward function with signature
            ``curve_func(q, fitted_params, prior_sampler, reflectivity_kwargs)``.
        return_param_errors:
            If ``True``, also return covariance-derived parameter standard errors.
        **kwargs:
            Additional keyword arguments forwarded to ``curve_fit``.
    """
    reflectivity_kwargs = reflectivity_kwargs or {}

    if bounds is not None:
        # introduce a small perturbation for fixed bounds
        epsilon = 1e-6
        adjusted_bounds = bounds.copy()

        for i in range(bounds.shape[1]): 
            if bounds[0, i] == bounds[1, i]:
                #adjusted_bounds[0, i] -= epsilon   #can create some issues when a bound is (0., 0.) for a param that should be nonnegative
                adjusted_bounds[1, i] += epsilon

        init_params = np.clip(init_params, *adjusted_bounds)
        if method != 'lm':
            kwargs['bounds'] = adjusted_bounds
    
    reflectivity_kwargs_torch = {}
    for key, value in reflectivity_kwargs.items():
        if isinstance(value, float):
            reflectivity_kwargs_torch[key] = torch.tensor([[value]], dtype=torch.float64)
        elif isinstance(value, np.ndarray):
            reflectivity_kwargs_torch[key] = torch.tensor(value, dtype=torch.float64).unsqueeze(0)
        else:
            reflectivity_kwargs_torch[key] = value

    curve = np.clip(curve, a_min=1e-12, a_max=None)

    if error_bars is not None and scale_curve_func == np.log10:
        error_bars = np.clip(error_bars, a_min=1e-20, a_max=None)
        scaled_error_bars = error_bars / (curve * np.log(10))
    else:
        scaled_error_bars = None

    if polishing_max_nfev is not None:
        if method == "lm":
            kwargs["maxfev"] = polishing_max_nfev
        else:
            kwargs["max_nfev"] = polishing_max_nfev

    res = curve_fit(
        f=get_scaled_curve_func(
            scale_curve_func=scale_curve_func,
            prior_sampler=prior_sampler,
            reflectivity_kwargs=reflectivity_kwargs_torch,
            curve_func=curve_func,
        ),
        xdata=q, 
        ydata=scale_curve_func(curve).reshape(-1), #.reshape(-1) ensures it will also work for with multichannel curves
        p0=init_params,
        sigma=scaled_error_bars,
        absolute_sigma=True,
        method=method,
        **kwargs
    )

    if curve_func is None:
        fitted_curve = prior_sampler.param_model.reflectivity(
            torch.tensor(q, dtype=torch.float64),
            torch.tensor(res[0], dtype=torch.float64).unsqueeze(0),
            **reflectivity_kwargs_torch
        ).squeeze().numpy()
    else:
        fitted_curve = curve_func(
            q=q,
            fitted_params=np.asarray(res[0]),
            prior_sampler=prior_sampler,
            reflectivity_kwargs=reflectivity_kwargs_torch,
        )

    if return_param_errors:
        covariance = res[1]
        if covariance is not None and np.ndim(covariance) == 2 and np.all(np.isfinite(covariance)):
            fitted_param_errors = np.sqrt(np.diag(covariance))
        else:
            fitted_param_errors = np.full_like(np.asarray(res[0]), np.nan, dtype=np.float64)
    else:
        fitted_param_errors = None

    return res[0], fitted_param_errors, fitted_curve


def get_scaled_curve_func(
    scale_curve_func=np.log10,
    prior_sampler: PriorSampler = None,
    reflectivity_kwargs: dict = None,
    curve_func=None,
):
    """
    Build a scaled forward function compatible with ``scipy.optimize.curve_fit``.

    Args:
        scale_curve_func:
            Scaling applied to the simulated reflectivity before comparing it to
            the scaled data.
        prior_sampler:
            Prior sampler whose parametric model defines the default reflectivity.
        reflectivity_kwargs:
            Optional extra keyword arguments passed to the reflectivity model.
        curve_func:
            Optional custom forward function with signature
            ``curve_func(q, fitted_params, prior_sampler, reflectivity_kwargs)``.

    Returns:
        A callable suitable for ``curve_fit``.
    """
    reflectivity_kwargs = reflectivity_kwargs or {}

    def scaled_curve_func(q, *fitted_params):
        fitted_params = np.asarray(fitted_params)

        if curve_func is None:
            q_tensor = torch.from_numpy(q).to(torch.float64)
            fitted_params_tensor = torch.tensor(fitted_params, dtype=torch.float64).unsqueeze(0)

            fitted_curve_tensor = prior_sampler.param_model.reflectivity(
                q_tensor,
                fitted_params_tensor,
                **reflectivity_kwargs
            )
            fitted_curve = fitted_curve_tensor.squeeze().numpy()
        else:
            fitted_curve = curve_func(
                q=q,
                fitted_params=fitted_params,
                prior_sampler=prior_sampler,
                reflectivity_kwargs=reflectivity_kwargs,
            )

        scaled_curve = scale_curve_func(fitted_curve)
        return scaled_curve.reshape(-1)

    return scaled_curve_func


def get_fit_with_growth(
    q: np.ndarray,
    curve: np.ndarray,
    init_params: np.ndarray,
    prior_sampler: PriorSampler,
    bounds: np.ndarray = None,
    init_d_change: float = 0.0,
    max_d_change: float = 30.0,
    d_idx: int = 0,
    scale_curve_func=np.log10,
    error_bars: np.ndarray = None,
    method: str = "trf",
    polishing_max_nfev: int = None,
    reflectivity_kwargs: dict = None,
    return_param_errors: bool = False,
    **kwargs
):
    """
    Fit a reflectivity curve while allowing one thickness parameter to change
    linearly across the acquisition.

    The fitted parameter vector is extended by one extra parameter ``delta_d``,
    representing the total thickness change over the scan.

    Args:
        q:
            Experimental q-values of shape ``(n_q,)``.
        curve:
            Experimental reflectivity values of shape ``(n_q,)``.
        init_params:
            Initial base parameter vector.
        prior_sampler:
            Prior sampler defining the parameterization and reflectivity model.
        bounds:
            Optional bounds of shape ``(2, n_params)`` for the base parameters.
        init_d_change:
            Initial value for the total thickness change.
        max_d_change:
            Upper bound for the total thickness change.
        d_idx:
            Index of the thickness parameter to modify.
        scale_curve_func:
            Scaling function applied before fitting.
        error_bars:
            Optional experimental uncertainties.
        method:
            Optimization method for ``curve_fit``.
        polishing_max_nfev:
            Maximum number of function evaluations.
        reflectivity_kwargs:
            Optional extra reflectivity keyword arguments.
        return_param_errors:
            If ``True``, also return covariance-derived parameter standard errors.
        **kwargs:
            Additional keyword arguments forwarded to ``refl_fit``.
    """
    init_params_growth = np.concatenate([np.asarray(init_params), np.array([init_d_change])])

    if bounds is not None:
        growth_bounds = np.concatenate(
            [bounds, np.array([[0.0], [max_d_change]])],
            axis=1,
        )
    else:
        growth_bounds = None

    result = refl_fit(
        q=q,
        curve=curve,
        init_params=init_params_growth,
        prior_sampler=prior_sampler,
        bounds=growth_bounds,
        error_bars=error_bars,
        scale_curve_func=scale_curve_func,
        method=method,
        polishing_max_nfev=polishing_max_nfev,
        reflectivity_kwargs=reflectivity_kwargs,
        curve_func=_growth_curve_func(d_idx=d_idx),
        return_param_errors=return_param_errors,
        **kwargs
    )

    params, param_errors, fitted_curve = result

    params = params.copy()
    params[d_idx] += params[-1] / 2.0

    return params, param_errors, fitted_curve

def _growth_curve_func(d_idx: int = 0):
    """
    Build a forward model for linear thickness growth during acquisition.

    The last fitted parameter is interpreted as the total thickness change
    ``delta_d`` across the scan.
    """
    def curve_func(
        q: np.ndarray,
        fitted_params: np.ndarray,
        prior_sampler: PriorSampler,
        reflectivity_kwargs: dict = None,
    ) -> np.ndarray:
        reflectivity_kwargs = reflectivity_kwargs or {}

        fitted_params = np.asarray(fitted_params, dtype=np.float64)
        base_params = fitted_params[:-1].copy()
        delta_d = float(fitted_params[-1])

        q_size = q.size
        d_shift = np.linspace(0.0, delta_d, q_size)

        theta = np.repeat(base_params[None, :], q_size, axis=0)
        theta[:, d_idx] = theta[:, d_idx] + d_shift

        q_tensor = torch.as_tensor(q, dtype=torch.float64).unsqueeze(-1)
        theta_tensor = torch.as_tensor(theta, dtype=torch.float64)

        fitted_curve_tensor = prior_sampler.param_model.reflectivity(
            q_tensor,
            theta_tensor,
            **reflectivity_kwargs
        )
        return fitted_curve_tensor.reshape(-1).cpu().numpy()

    return curve_func



def batch_refl_fit(
        q: np.ndarray, 
        curves: np.ndarray,
        init_params: np.ndarray, # (n_curves, n_params)
        prior_sampler: PriorSampler,
        bounds: np.ndarray = None,
        error_bars: np.ndarray = None,
        scale_curve_func=np.log10,
        method: str = 'trf', #'lm', 'trf'
        polishing_max_steps: int = None,
        reflectivity_kwargs: dict = None,
        n_jobs: int = -1,
        verbose: int = 5,
        **kwargs
):
    """
    Fit (polished fit) multiple reflectivity curves in parallel using joblib.

    Parameters
    ----------
    q : np.ndarray
        1D array of momentum transfer values (same for all curves).
    curves : np.ndarray
        2D array of reflectivity curves with shape (n_curves, n_q).
    init_params : np.ndarray
        2D array of initial parameter guesses (n_curves, n_params).
    prior_sampler : PriorSampler
        The prior sampler.
    bounds : np.ndarray, optional
        Bounds for the parameters, shape (2, n_params). Shared by all the curves. Default: None.
    error_bars : np.ndarray, optional
        Error bars for the curves, shape (n_curves, n_q). Default: None.
    scale_curve_func : callable, optional
        Function to scale the curves. Default: `np.log10`.
    method : str, optional
        The method to use for the fitting. Default: 'trf'.
    polishing_max_steps : int, optional
        The maximum number of function evaluations for the polishing step. Default: None.
    reflectivity_kwargs : dict, optional
        Keyword arguments for the reflectivity function. Default: None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default: -1 (all CPUs).
    verbose : int, optional
        The verbosity level for joblib. Default: 5.
    **kwargs : dict
        Extra keyword arguments passed to `scipy.optimize.curve_fit`.

    Returns
    -------
    params_array : np.ndarray
        Array of fitted parameter values for each curve, shape (n_curves, n_params).
    error_bars_array : np.ndarray
        Array of error bars for the fitted parameter values, shape (n_curves, n_params).
    curves_array : np.ndarray
        Array of fitted reflectivity curves, shape (n_curves, n_q).
    """
    if bounds is not None:
        if bounds.ndim == 2:
            bounds = bounds[None].repeat(curves.shape[0], 0)
        elif bounds.ndim == 3:
            assert bounds.shape[0] == curves.shape[0], f"Bounds must have the same number of curves as the number of curves, got {bounds.shape[0]} and {curves.shape[0]}"
        else:
            raise ValueError(f"Bounds must be a 2D or 3D array, got {bounds.ndim}D array")
    else:
        bounds = [None] * curves.shape[0]
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(refl_fit)(
            q=q, curve=curve, init_params=init_params,
            bounds=bound,
            prior_sampler=prior_sampler,
            error_bars=error_bars,
            method=method,
            scale_curve_func=scale_curve_func,
            polishing_max_steps=polishing_max_steps,
            reflectivity_kwargs=reflectivity_kwargs,
            **kwargs
        )
        for curve, init_params, bound in zip(curves, init_params, bounds)
    )

    params_array, error_bars, curves_array = zip(*results)
    return np.array(params_array), np.array(error_bars), np.array(curves_array)
