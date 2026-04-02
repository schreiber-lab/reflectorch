from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from reflectorch.inference.plotting import plot_reflectivity


ArrayLike1D = np.ndarray
DQType = Optional[Union[float, np.ndarray]]


@dataclass
class ReflectivityData:
    """
    Experimental reflectivity data for a single-channel observation.

    Attributes
    ----------
    q
        q-grid of shape ``(n_q,)``.
    R
        Reflectivity values of shape ``(n_q,)``.
    dR
        Optional uncertainties of shape ``(n_q,)``.
    dq
        Optional q-resolution information. May be:
        - ``None``
        - a scalar
        - an array of shape ``(n_q,)``
    """
    q: np.ndarray
    R: np.ndarray
    dR: Optional[np.ndarray] = None
    dq: DQType = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q)
        self.R = np.asarray(self.R)

        if self.q.ndim != 1:
            raise ValueError(f"`q` must have shape (n_q,), got {self.q.shape}")
        if self.R.ndim != 1:
            raise ValueError(f"`R` must have shape (n_q,), got {self.R.shape}")
        if self.q.shape[0] != self.R.shape[0]:
            raise ValueError(
                f"`q` and `R` must have the same length, got {self.q.shape[0]} and {self.R.shape[0]}"
            )

        if self.dR is not None:
            self.dR = np.asarray(self.dR)
            if self.dR.ndim != 1:
                raise ValueError(f"`dR` must have shape (n_q,), got {self.dR.shape}")
            if self.dR.shape[0] != self.q.shape[0]:
                raise ValueError(
                    f"`dR` and `q` must have the same length, got {self.dR.shape[0]} and {self.q.shape[0]}"
                )

        if self.dq is not None and not np.isscalar(self.dq):
            self.dq = np.asarray(self.dq)
            if self.dq.ndim != 1:
                raise ValueError(f"`dq` must be scalar or shape (n_q,), got {self.dq.shape}")
            if self.dq.shape[0] != self.q.shape[0]:
                raise ValueError(
                    f"`dq` and `q` must have the same length, got {self.dq.shape[0]} and {self.q.shape[0]}"
                )

    def plot(self, **kwargs):
        return plot_reflectivity(q_exp=self.q, r_exp=self.R, yerr=self.dR, **kwargs)