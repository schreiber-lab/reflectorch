try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from numpy import ndarray

NORMALIZE_MODE = Literal["first", "max", "incoming_intensity"]


def intensity2reflectivity(intensity: ndarray, mode: NORMALIZE_MODE, incoming_intensity=None) -> np.ndarray:
    if mode == "first":
        return intensity / intensity[0]
    if mode == "max":
        return intensity / intensity.max()
    if mode == "incoming_intensity":
        if incoming_intensity is None:
            raise ValueError("incoming_intensity is None")
        return intensity / incoming_intensity
    raise ValueError(f"Unknown mode {mode}")
