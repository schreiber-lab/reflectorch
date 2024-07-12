from dataclasses import dataclass

import numpy as np

from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.inference.preprocess_exp.footprint import apply_footprint_correction, BEAM_SHAPE
from reflectorch.inference.preprocess_exp.normalize import intensity2reflectivity, NORMALIZE_MODE
from reflectorch.inference.preprocess_exp.attenuation import apply_attenuation_correction
from reflectorch.inference.preprocess_exp.cut_with_q_ratio import cut_curve
from reflectorch.utils import angle_to_q


def standard_preprocessing(
        intensity: np.ndarray,
        scattering_angle: np.ndarray,
        attenuation: np.ndarray,
        q_interp: np.ndarray,
        wavelength: float,
        beam_width: float,
        sample_length: float,
        min_intensity: float = 1e-10,
        beam_shape: BEAM_SHAPE = "gauss",
        normalize_mode: NORMALIZE_MODE = "max",
        incoming_intensity: float = None,
        max_q: float = None,  # if provided, max_angle is ignored
        max_angle: float = None,
) -> dict:
    """Preprocesses a raw experimental reflectivity curve by applying attenuation correction, footprint correction, cutting at a maximum q value and interpolation

    Args:
        intensity (np.ndarray): array of intensities of the reflectivity curve
        scattering_angle (np.ndarray): array of scattering angles
        attenuation (np.ndarray): attenuation factors for each measured point
        q_interp (np.ndarray): reciprocal space points used for the interpolation
        wavelength (float): the wavelength of the beam
        beam_width (float): the beam width
        sample_length (float): the sample length
        min_intensity (float, optional): intensities lower than this value are removed. Defaults to 1e-10.
        beam_shape (BEAM_SHAPE, optional): the shape of the beam, either "gauss" or "box". Defaults to "gauss".
        normalize_mode (NORMALIZE_MODE, optional): normalization mode, either "first", "max" or "incoming_intensity". Defaults to "max".
        incoming_intensity (float, optional): array of intensities for the "incoming_intensity" normalization. Defaults to None.
        max_q (float, optional): the maximum q value at which the curve is cut. Defaults to None.
        max_angle (float, optional): the maximum scattering angle at which the curve is cut; only used if max_q is not provided. Defaults to None.

    Returns:
        dict: dictionary containing the interpolated reflectivity curve, the curve before interpolation, the q values before interpolation, the q values after interpolation and the q ratio of the cutting
    """
    intensity = apply_attenuation_correction(
        intensity,
        attenuation,
        scattering_angle,
    )

    intensity = apply_footprint_correction(
        intensity, scattering_angle, beam_width=beam_width, sample_length=sample_length, beam_shape=beam_shape
    )

    curve = intensity2reflectivity(intensity, normalize_mode, incoming_intensity)

    curve, scattering_angle = remove_low_statistics(curve, scattering_angle, thresh=min_intensity)

    q = angle_to_q(scattering_angle, wavelength)

    q, curve, q_ratio = cut_curve(q, curve, max_q, max_angle, wavelength)

    curve_interp = interp_reflectivity(q_interp, q, curve)

    assert np.all(np.isfinite(curve_interp))
    assert np.all(np.isfinite(curve))
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(q_interp))
    assert np.all(curve > 0.)
    assert np.all(curve_interp > 0.)

    return {
        "curve_interp": curve_interp, "curve": curve, "q_values": q, "q_interp": q_interp, "q_ratio": q_ratio,
    }


def remove_low_statistics(curve, scattering_angle, thresh: float = 1e-7):
    indices = (curve > thresh) & np.isfinite(curve)
    return curve[indices], scattering_angle[indices]


@dataclass
class StandardPreprocessing:
    q_interp: np.ndarray = None
    wavelength: float = 1.
    beam_width: float = None
    sample_length: float = None
    beam_shape: BEAM_SHAPE = "gauss"
    normalize_mode: NORMALIZE_MODE = "max"
    incoming_intensity: float = None

    def preprocess(self,
                   intensity: np.ndarray,
                   scattering_angle: np.ndarray,
                   attenuation: np.ndarray,
                   **kwargs
                   ) -> dict:
        attrs = self._get_updated_attrs(**kwargs)
        return standard_preprocessing(
            intensity,
            scattering_angle,
            attenuation,
            **attrs
        )

    __call__ = preprocess

    def set_parameters(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.__annotations__:
                setattr(self, k, v)
            else:
                raise KeyError(f'Unknown parameter {k}.')

    def _get_updated_attrs(self, **kwargs):
        current_attrs = {k: getattr(self, k) for k in self.__annotations__.keys()}
        current_attrs.update(kwargs)
        return current_attrs
