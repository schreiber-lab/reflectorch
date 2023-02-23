from dataclasses import dataclass

import numpy as np

from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.inference.preprocess_exp.footprint import apply_footprint_correction, BEAM_SHAPE
from reflectorch.inference.preprocess_exp.normalize import intensity2reflectivity, NORMALIZE_MODE
from reflectorch.inference.preprocess_exp.attenuation import apply_attenuation_correction
from reflectorch.inference.preprocess_exp.utils import angle2q


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

    q = angle2q(scattering_angle, wavelength)

    curve_interp = interp_reflectivity(q_interp, q, curve)

    return {
        'curve_interp': curve_interp, 'curve': curve,
    }


@dataclass
class StandardPreprocessing:
    q_interp: np.ndarray
    wavelength: float
    beam_width: float
    sample_length: float
    beam_shape: BEAM_SHAPE = "gauss",
    normalize_mode: NORMALIZE_MODE = "max"

    def preprocess(self,
                   intensity: np.ndarray,
                   scattering_angle: np.ndarray,
                   attenuation: np.ndarray,
                   **kwargs
                   ) -> dict:
        return standard_preprocessing(
            intensity,
            scattering_angle,
            attenuation,
            q_interp=self.q_interp,
            wavelength=self.wavelength,
            beam_width=self.beam_width,
            sample_length=self.sample_length,
            beam_shape=self.beam_shape,
            normalize_mode=self.normalize_mode,
            **kwargs
        )
