import numpy as np


def interp_reflectivity(q_interp, q, reflectivity, min_value: float = 1e-10):
    """Interpolate data on a base 10 logarithmic scale

    Args:
        q_interp (array-like): reciprocal space points used for the interpolation
        q (array-like): reciprocal space points of the measured reflectivity curve
        reflectivity (array-like): reflectivity curve measured at the points ``q``
        min_value (float, optional): minimum intensity of the reflectivity curve. Defaults to 1e-10.

    Returns:
        array-like: interpolated reflectivity curve
    """
    return 10 ** np.interp(q_interp, q, np.log10(np.clip(reflectivity, min_value, None)))
