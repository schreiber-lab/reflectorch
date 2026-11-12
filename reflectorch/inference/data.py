from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from reflectorch.inference.plotting import plot_reflectivity


ArrayLike1D = np.ndarray
DQType = Optional[Union[float, np.ndarray]]


@dataclass
class ReflectivityData:
    """
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


@dataclass
class MultiChannelReflectivityData:
    """
    """
    q: np.ndarray
    R: np.ndarray
    dR: Optional[np.ndarray] = None
    dq: DQType = None

    q_raw_list: Optional[List[np.ndarray]] = None
    R_raw_list: Optional[List[np.ndarray]] = None
    dR_raw_list: Optional[List[Optional[np.ndarray]]] = None
    dq_raw_list: Optional[List[Optional[Union[float, np.ndarray]]]] = None

    channel_labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q)
        self.R = np.asarray(self.R)

        if self.q.ndim != 1:
            raise ValueError(f"`q` must have shape (n_q,), got {self.q.shape}")
        if self.R.ndim != 2:
            raise ValueError(f"`R` must have shape (n_channels, n_q), got {self.R.shape}")
        if self.R.shape[1] != self.q.shape[0]:
            raise ValueError(
                f"`R.shape[1]` must equal `len(q)`, got {self.R.shape[1]} and {self.q.shape[0]}"
            )

        n_channels = self.R.shape[0]
        n_q = self.q.shape[0]

        if self.dR is not None:
            self.dR = np.asarray(self.dR)
            if self.dR.shape != (n_channels, n_q):
                raise ValueError(
                    f"`dR` must have shape {(n_channels, n_q)}, got {self.dR.shape}"
                )

        if self.dq is not None and not np.isscalar(self.dq):
            self.dq = np.asarray(self.dq)
            if self.dq.ndim != 1:
                raise ValueError(f"`dq` must be scalar or shape (n_q,), got {self.dq.shape}")
            if self.dq.shape[0] != n_q:
                raise ValueError(
                    f"`dq` and `q` must have the same length, got {self.dq.shape[0]} and {n_q}"
                )

        if self.channel_labels is None:
            self.channel_labels = [f"channel {i}" for i in range(n_channels)]
        else:
            self.channel_labels = [str(x) for x in self.channel_labels]
            if len(self.channel_labels) != n_channels:
                raise ValueError(
                    f"`channel_labels` must have length {n_channels}, got {len(self.channel_labels)}"
                )

        if self.q_raw_list is not None and len(self.q_raw_list) != n_channels:
            raise ValueError(
                f"`q_raw_list` must have length {n_channels}, got {len(self.q_raw_list)}"
            )
        if self.R_raw_list is not None and len(self.R_raw_list) != n_channels:
            raise ValueError(
                f"`R_raw_list` must have length {n_channels}, got {len(self.R_raw_list)}"
            )
        if self.dR_raw_list is not None and len(self.dR_raw_list) != n_channels:
            raise ValueError(
                f"`dR_raw_list` must have length {n_channels}, got {len(self.dR_raw_list)}"
            )
        if self.dq_raw_list is not None and len(self.dq_raw_list) != n_channels:
            raise ValueError(
                f"`dq_raw_list` must have length {n_channels}, got {len(self.dq_raw_list)}"
            )