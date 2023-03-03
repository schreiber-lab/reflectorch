import numpy as np


def interp_reflectivity(q_interp, q, reflectivity, min_value: float = 1e-10):
    """Interpolate data on a base10 logarithmic scale."""
    return 10 ** np.interp(q_interp, q, np.log10(np.clip(reflectivity, min_value, None)))
