import warnings

import numpy as np
from scipy.optimize import minimize, curve_fit

from reflectorch.data_generation.reflectivity import abeles_np


__all__ = [
    "standard_refl_fit",
    "fit_refl_curve",
    "restore_masked_params",
]


def standard_restore_params(fitted_params) -> dict:
    num_layers = (fitted_params.size - 2) // 3

    return dict(
        thickness=fitted_params[:num_layers],
        roughness=fitted_params[num_layers:2 * num_layers + 1],
        sld=fitted_params[2 * num_layers + 1:],
    )


def mse_loss(curve1, curve2):
    return np.sum((curve1 - curve2) ** 2)


def standard_refl_fit(
        q: np.ndarray, curve: np.ndarray,
        init_params: np.ndarray,
        bounds: np.ndarray = None,
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
        **kwargs
) -> np.ndarray:
    if bounds is not None:
        kwargs['bounds'] = bounds

    res = curve_fit(
        get_scaled_curve_func(
            refl_generator=refl_generator,
            restore_params_func=restore_params_func,
            scale_curve_func=scale_curve_func,
        ),
        q, scale_curve_func(curve),
        p0=init_params, **kwargs
    )

    return res[0]


def fit_refl_curve(q: np.ndarray, curve: np.ndarray,
                   init_params: np.ndarray,
                   bounds: np.ndarray = None,
                   refl_generator=abeles_np,
                   restore_params_func=standard_restore_params,
                   scale_curve_func=np.log10,
                   **kwargs
                   ) -> np.ndarray:
    fitting_func = get_fitting_func(
        q=q, curve=curve,
        refl_generator=refl_generator,
        restore_params_func=restore_params_func,
        scale_curve_func=scale_curve_func,
    )

    res = minimize(fitting_func, init_params, bounds=bounds, **kwargs)

    if not res.success:
        warnings.warn(f"Minimization did not converge.")
    return res.x


def get_scaled_curve_func(
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
):
    def scaled_curve_func(q, *fitted_params):
        fitted_params = restore_params_func(np.asarray(fitted_params))
        fitted_curve = refl_generator(q, **fitted_params)
        scaled_curve = scale_curve_func(fitted_curve)
        return scaled_curve

    return scaled_curve_func


def get_fitting_func(
        q: np.ndarray,
        curve: np.ndarray,
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
        loss_func=mse_loss,
):
    scaled_curve = scale_curve_func(curve)

    def fitting_func(fitted_params):
        fitted_params = restore_params_func(fitted_params)
        fitted_curve = refl_generator(q, **fitted_params)
        loss = loss_func(scale_curve_func(fitted_curve), scaled_curve)
        return loss

    return fitting_func


def restore_masked_params(fixed_params, fixed_mask):
    def restore_params(fitted_params) -> dict:
        params = np.empty_like(fixed_mask).astype(fitted_params.dtype)
        params[fixed_mask] = fixed_params
        params[~fixed_mask] = fitted_params
        return standard_restore_params(params)

    return restore_params
