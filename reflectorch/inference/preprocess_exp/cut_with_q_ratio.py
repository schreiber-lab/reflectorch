import numpy as np

from reflectorch.utils import angle_to_q


def cut_curve(q: np.ndarray, curve: np.ndarray, max_q: float, max_angle: float, wavelength: float):
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
