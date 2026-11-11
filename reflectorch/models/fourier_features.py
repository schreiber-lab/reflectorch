import torch
import torch.nn as nn
from torch import Tensor
import math

class IdentityTimeEmbedding(nn.Module):
    """Returns time as a (B,1) feature."""
    def __init__(self):
        super().__init__()
        self.out_features = 1

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            return t.view(1, 1)
        if t.ndim == 1:
            return t[:, None]
        if t.ndim == 2 and t.shape[-1] == 1:
            return t
        raise ValueError(f"Expected t shape (), (B,), or (B,1); got {tuple(t.shape)}")


class DeterministicFourierFeatures(nn.Module):
    """
    Deterministic Fourier features.
    For scalar time input (D=1) output dim = 2*n_freq.
    """
    def __init__(
        self,
        n_freq: int,
        *,
        scale: float = 2 * math.pi,
        ff_type: str = "linear",
        half_shift: bool = False,
    ):
        super().__init__()
        if n_freq <= 0:
            raise ValueError("n_freq must be > 0")
        self.n_freq = n_freq
        self.scale = scale
        self.schedule = ff_type
        self.half_shift = half_shift

        if ff_type == "powers_of_two":
            freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float32)
        elif ff_type == "linear":
            start = 0.5 if half_shift else 1.0
            freqs = torch.arange(start, start + n_freq, 1.0, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown ff_type={ff_type}")

        self.register_buffer("freqs", freqs)

        self.out_features = 2 * n_freq

    def forward(self, x: Tensor) -> Tensor:
        # Accept (B,), (B,1), (B,D)
        if x.dim() == 1:
            x = x[:, None]
        elif x.dim() != 2:
            raise ValueError(f"Expected x with shape (B,), (B,1) or (B,D), got {tuple(x.shape)}")

        B, D = x.shape
        x_freq = self.scale * torch.einsum("f,bd->bfd", self.freqs, x).reshape(B, self.n_freq * D)
        return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)


class SinusoidalPositionalTimeEmbedding(nn.Module):
    """Vaswani-style sinusoidal embedding (log-spaced frequencies)."""
    def __init__(self, embed_dim: int = 32, max_freq: float = 1000.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(max_freq) / embed_dim)
        )
        self.register_buffer("div_term", div_term)
        self.out_features = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t[:, None]
        elif not (t.ndim == 2 and t.shape[-1] == 1):
            raise ValueError(f"Expected t shape (), (B,), or (B,1); got {tuple(t.shape)}")

        B = t.shape[0]
        out = torch.zeros(B, self.embed_dim, device=t.device, dtype=t.dtype)
        out[:, 0::2] = torch.sin(t * self.div_term)
        out[:, 1::2] = torch.cos(t * self.div_term)
        return out


class RandomFourierTimeEmbedding(nn.Module):
    """Random Fourier features for time, with optional learnable frequencies."""
    def __init__(self, embed_dim: int = 32, scale: float = 30.0, learnable: bool = True):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        self.embed_dim = embed_dim
        self.scale = scale
        W = torch.randn(embed_dim // 2) * scale
        if learnable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer("W", W)
        self.out_features = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.view(1)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        elif t.ndim != 1:
            raise ValueError(f"Expected t shape (), (B,), or (B,1); got {tuple(t.shape)}")

        proj = t[:, None] * self.W[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)