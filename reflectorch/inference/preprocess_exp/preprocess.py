from dataclasses import dataclass

import numpy as np

from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.inference.preprocess_exp.footprint import apply_footprint_correction, BEAM_SHAPE
from reflectorch.inference.preprocess_exp.normalize import intensity2reflectivity, NORMALIZE_MODE
from reflectorch.inference.preprocess_exp.attenuation import apply_attenuation_correction
from reflectorch.utils import angle_to_q


def standard_preprocessing(
        intensity: np.ndarray,
        scattering_angle: np.ndarray,
        attenuation: np.ndarray,
        q_interp: np.ndarray,
        wavelength: float,
        beam_width: float,
        sample_length: float,
        beam_shape: BEAM_SHAPE = "gauss",
        normalize_mode: NORMALIZE_MODE = "max",
        incoming_intensity: float = None,
) -> dict:
    intensity = apply_attenuation_correction(
        intensity,
        attenuation,
        scattering_angle,
    )

    intensity = apply_footprint_correction(
        intensity, scattering_angle, beam_width=beam_width, sample_length=sample_length, beam_shape=beam_shape
    )

    curve = intensity2reflectivity(intensity, normalize_mode, incoming_intensity)

    q = angle_to_q(scattering_angle, wavelength)

    curve_interp = interp_reflectivity(q_interp, q, curve)

    return {
        "curve_interp": curve_interp, "curve": curve, "q_values": q, "q_interp": q_interp,
    }


@dataclass
class StandardPreprocessing:
    q_interp: np.ndarray = None
    wavelength: float = 1.
    beam_width: float = None
    sample_length: float = None
    beam_shape: BEAM_SHAPE = "gauss"
    normalize_mode: NORMALIZE_MODE = "max"

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
