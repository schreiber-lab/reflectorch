import numpy as np


def angle2q(scattering_angle: np.ndarray, wavelength: float):
    return 4 * np.pi / wavelength * np.sin(scattering_angle / 2 * np.pi / 180)
