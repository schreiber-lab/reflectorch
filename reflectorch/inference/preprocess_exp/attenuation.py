import numpy as np


def apply_attenuation_correction(
        intensity: np.ndarray,
        attenuation: np.ndarray,
        scattering_angle: np.ndarray = None,
        correct_discontinuities: bool = True
) -> np.ndarray:
    """Applies attenuation correction to experimental reflectivity curves

    Args:
        intensity (np.ndarray): intensities of an experimental reflectivity curve
        attenuation (np.ndarray): attenuation factors for each measured point
        scattering_angle (np.ndarray, optional): scattering angles of the measured points. Defaults to None.
        correct_discontinuities (bool, optional): whether to correct discontinuities in the measured curves. Defaults to True.

    Returns:
        np.ndarray: the corrected reflectivity curve
    """
    intensity = intensity / attenuation
    if correct_discontinuities:
        if scattering_angle is None:
            raise ValueError("correct_discontinuities options requires scattering_angle, but scattering_angle is None.")
        intensity = apply_discontinuities_correction(intensity, scattering_angle)
    return intensity


def apply_discontinuities_correction(intensity: np.ndarray, scattering_angle: np.ndarray) -> np.ndarray:
    intensity = intensity.copy()
    diff_angle = np.diff(scattering_angle)
    for i in range(len(diff_angle)):
        if diff_angle[i] == 0:
            factor = intensity[i] / intensity[i + 1]
            intensity[(i + 1):] *= factor
    return intensity
