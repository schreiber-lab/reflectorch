import numpy as np

from reflectorch.utils import angle_to_q


def cut_curve(q: np.ndarray, curve: np.ndarray, max_q: float, max_angle: float, wavelength: float):
    """Cuts an experimental reflectivity curve at a maximum q position

    Args:
        q (np.ndarray): the array of q points
        curve (np.ndarray): the experimental reflectivity curve
        max_q (float): the maximum q value at which the curve is cut
        max_angle (float): the maximum scattering angle at which the curve is cut; only used if max_q is not provided
        wavelength (float): the wavelength of the beam

    Returns:
        tuple: the q array after cutting, the reflectivity curve after cutting, and the ratio between the maximum q after cutting and before cutting
    """
    if max_angle is None and max_q is None:
        q_ratio = 1.
    else:
        if max_q is None:
            max_q = angle_to_q(max_angle, wavelength)

        q_ratio = max_q / q.max()

        if q_ratio < 1.:
            idx = np.argmax(q > max_q)
            q = q[:idx] / q_ratio
            curve = curve[:idx]
    return q, curve, q_ratio
