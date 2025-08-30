from __future__ import annotations
from typing import Union

import torch
from torch import nn, Tensor, stack, cat
from reflectorch.models.activations import activation_by_name
import reflectorch

###embedding network adapted from the PANPE repository

__all__ = [
    "IntegralConvEmbedding",
]

class IntegralConvEmbedding(nn.Module):
    def __init__(
        self,
        z_num: Union[int, tuple[int, ...]],
        z_range: tuple[float, float] = None,
        in_dim: int = 2,
        kernel_coef: int = 16,
        dim_embedding: int = 256,
        conv_dims: tuple[int, ...] = (32, 64, 128),
        num_blocks: int = 4,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_fft: bool = False,
        activation: str = "gelu",
        conv_activation: str = "lrelu",
        resnet_activation: str = "relu",
    ) -> None:
        super().__init__()

        if isinstance(z_num, int):
            z_num = (z_num,)
        num_kernel = len(z_num)

        if z_range is not None:
            zs = [(z_range[0], z_range[1], nz) for nz in z_num]
        else:
            zs = z_num

        self.in_dim = in_dim

        self.kernels = nn.ModuleList(
            [
                IntegralKernelBlock(
                    z,
                    in_dim,
                    kernel_coef=kernel_coef,
                    latent_dim=dim_embedding,
                    conv_dims=conv_dims,
                    use_fft=use_fft,
                    activation=activation,
                    conv_activation=conv_activation,
                )
                for z in zs
            ]
        )

        self.fc = reflectorch.models.networks.residual_net.ResidualMLP(
            dim_in=dim_embedding * num_kernel, 
            dim_out=dim_embedding,
            layer_width=2 * dim_embedding,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation=resnet_activation,
        )

    def forward(self, q, y, drop_mask=None) -> Tensor:
        x = cat([kernel(q, y, drop_mask=drop_mask) for kernel in self.kernels], dim=-1)
        x = self.fc(x)
        
        return x


class IntegralKernelBlock(nn.Module):
    """
    Examples:
        >>> x = torch.rand(2, 100)
        >>> y = torch.rand(2, 100, 3)
        >>> block = IntegralKernelBlock((0, 1, 10), in_dim=3,  latent_dim=32)
        >>> output = block(x, y)
        >>> output.shape
        torch.Size([2, 32])

        >>> block = IntegralKernelBlock(10, in_dim=3,  latent_dim=32)
        >>> output = block(x, y)
        >>> output.shape
        torch.Size([2, 32])
    """

    def __init__(
        self,
        z: tuple[float, float, int] or int,
        in_dim: int,
        kernel_coef: int = 2,
        latent_dim: int = 32,
        conv_dims: tuple[int, ...] = (32, 64, 128),
        use_fft: bool = False,
        activation: str = "gelu",
        conv_activation: str = "lrelu",
    ):
        super().__init__()

        if isinstance(z, int):
            z_num = z
            kernel = FullIntegralKernel(z_num, in_dim=in_dim, kernel_coef=kernel_coef)
        else:
            kernel = FastIntegralKernel(
                z, in_dim=in_dim, kernel_coef=kernel_coef, activation=activation
            )
            z_num = z[-1]

        assert z_num % 2 == 0, "z_num should be even"

        self.kernel = kernel
        self.z_num = z_num
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_fft = use_fft

        self.fc_in_dim = self.latent_dim + self.in_dim * self.z_num
        if self.use_fft:
            self.fc_in_dim += self.in_dim * 2 + self.in_dim * self.z_num

        self.conv = reflectorch.models.encoders.conv_encoder.ConvEncoder(
            dim_avpool=8,
            hidden_channels=conv_dims,
            in_channels=in_dim,
            dim_embedding=latent_dim,
            activation=conv_activation,
        )
        self.fc = FCBlock(
            in_dim=self.fc_in_dim, hid_dim=self.latent_dim * 2, out_dim=self.latent_dim
        )

    def forward(self, x: Tensor, y: Tensor, drop_mask: Tensor = None) -> Tensor:
        x = self.kernel(x, y, drop_mask=drop_mask)

        assert x.shape == (x.shape[0], self.in_dim, self.z_num)

        xc = self.conv(x)  # (batch, latent_dim)

        assert xc.shape == (x.shape[0], self.latent_dim)

        if self.use_fft:
            fft_x = torch.fft.rfft(x, dim=-1, norm="ortho")  # (batch, in_dim, z_num)

            fft_x = torch.cat(
                [fft_x.real, fft_x.imag], -1
            )  # (batch, in_dim, 2 * z_num)

            assert fft_x.shape == (x.shape[0], x.shape[1], self.z_num + 2)

            fft_x = fft_x.flatten(1)  # (batch, in_dim * (z_num + 2))

            x = torch.cat(
                [x.flatten(1), fft_x, xc], -1
            )  # (batch, in_dim * z_num * 3 + latent_dim)
        else:
            x = torch.cat([x.flatten(1), xc], -1)

        assert (
            x.shape[1] == self.fc_in_dim
        ), f"Expected dim {self.fc_in_dim}, got {x.shape[1]}"

        x = self.fc(x)  # (batch, latent_dim)

        return x


class FastIntegralKernel(nn.Module):
    def __init__(
        self,
        z: tuple[float, float, int],
        kernel_coef: int = 16,
        in_dim: int = 1,
        activation: str = "gelu",
    ):
        super().__init__()

        z = torch.linspace(*z)

        self.kernel = FCBlock(
            in_dim + 2, kernel_coef * in_dim, in_dim, activation=activation
        )

        self.register_buffer("z", z)

    def _get_z(self, x: Tensor):
        # x.shape == (batch_size, num_x)
        dz = self.z[1] - self.z[0]
        indices = torch.ceil((x - self.z[0] - dz / 2) / dz).to(torch.int64)

        z = torch.index_select(self.z, 0, indices.flatten()).view(*x.shape)

        return z, indices

    def forward(self, x: Tensor, y: Tensor, drop_mask=None):
        z, indices = self._get_z(x)
        xz = torch.stack([x, z], -1)
        kernel_input = torch.cat([xz, y], -1)
        output = self.kernel(kernel_input)  # (batch, x_num, in_dim)

        output = compute_means(
            output * y, indices, self.z.shape[-1], drop_mask=drop_mask
        )  # (batch, z_num, in_dim)

        output = output.swapaxes(1, 2)  # (batch, in_dim, z_num)

        return output


class FullIntegralKernel(nn.Module):
    def __init__(
        self,
        z_num: int,
        kernel_coef: int = 1,
        in_dim: int = 1,
    ):
        super().__init__()

        self.z_num = z_num
        self.in_dim = in_dim

        self.kernel = nn.Sequential(
            nn.Linear(in_dim + 1, z_num * kernel_coef),
            nn.LayerNorm(z_num * kernel_coef),
            nn.ReLU(),
            nn.Linear(z_num * kernel_coef, z_num * in_dim),
        )

    def forward(self, x: Tensor, y: Tensor, drop_mask=None):
        # x.shape == (batch_size, num_x)
        # y.shape == (batch_size, num_x, in_dim)
        # drop_mask.shape == (batch_size, num_x)

        batch_size, num_x = x.shape

        kernel_input = torch.cat([x.unsqueeze(-1), y], -1)  # (batch, x_num, in_dim + 1)
        x = self.kernel(kernel_input)  # (batch, x_num, z_num * in_dim)
        x = x.reshape(
            *x.shape[:-1], self.z_num, self.in_dim
        )  # (batch, x_num, z_num, in_dim)
        # permute to get (batch, z_num, x_num, in_dim)
        x = x.permute(0, 2, 1, 3)

        y = y.unsqueeze(1)  # (batch, 1, x_num, in_dim)

        assert x.shape == (
            batch_size,
            self.z_num,
            num_x,
            self.in_dim,
        )  # (batch, z_num, in_dim, x_num)
        assert y.shape == (
            batch_size,
            1,
            num_x,
            self.in_dim,
        )  # (batch, 1, x_num, in_dim)

        if drop_mask is not None:
            x = x * y
            x = x.permute(0, 2, 1, 3)  # (batch, x_num, z_num, in_dim)
            x = masked_mean(x, drop_mask)
        else:
            x = (x * y).mean(-2)  # (batch, z_num, in_dim)

        assert x.shape == (batch_size, self.z_num, self.in_dim), f"{x.shape}"

        x = x.swapaxes(1, 2)  # (batch, in_dim, z_num)

        return x


class FCBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hid_dim: int = 16,
        out_dim: int = 16,
        activation: str = "gelu",
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.activation = activation_by_name(activation)()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
        # return self.kernel(x)


def compute_means(x, indices, z: int, drop_mask: Tensor = None):
    """
    Compute the mean values of tensor 'x' for each unique index in 'indices' across each batch.

    This function calculates the mean of elements in 'x' that correspond to each unique index in 'indices'.
    The computation is performed for each batch separately, and the function is optimized to avoid Python loops
    by using advanced PyTorch operations.

    Parameters:
    x (torch.Tensor): A tensor of shape (batch_size, n, d) containing the values to be averaged.
                      'x' should be a floating-point tensor.
    indices (torch.Tensor): An integer tensor of shape (batch_size, n) containing the indices.
                            The values in 'indices' should be in the range [0, z-1].
    z (int): The number of unique indices. This determines the second dimension of the output tensor.
    drop_mask (torch.Tensor): A boolean tensor of shape (batch_size, n) containing a mask for the indices to drop.
                              If None, all indices are used.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, z, d) containing the mean values for each index in each batch.
                  If an index does not appear in a batch, its corresponding mean values are zeros.

    Example:
    >>> batch_size, n, d, z = 3, 4, 5, 6
    >>> indices = torch.randint(0, z, (batch_size, n))
    >>> x = torch.randn(batch_size, n, d)
    >>> y = compute_means(x, indices, z)
    >>> print(y.shape)
    torch.Size([3, 6, 5])
    """

    batch_size, n, d = x.shape
    device = x.device

    drop = drop_mask is not None

    # Initialize tensors to hold sums and counts
    sums = torch.zeros(batch_size, z + int(drop), d, device=device)
    counts = torch.zeros(batch_size, z + int(drop), device=device)

    if drop_mask is not None:
        # Set the values of the indices to drop to z
        indices = indices.masked_fill(~drop_mask, z)

    indices_expanded = indices.unsqueeze(-1).expand_as(x)
    sums.scatter_add_(1, indices_expanded, x)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=x.dtype))

    if drop:
        # Remove the z values from the sums and counts
        sums = sums[:, :-1]
        counts = counts[:, :-1]

    # Compute the mean and handle division by zero
    mean = sums / counts.unsqueeze(-1).clamp(min=1)

    return mean


def masked_mean(x, mask):
    """
    Computes the mean of tensor x along the x_size dimension,
    while masking out elements where the corresponding value in the mask is False.

    Args:
    x (torch.Tensor): A tensor of shape (batch, x_size, z, d).
    mask (torch.Tensor): A boolean mask of shape (batch, x_size).

    Returns:
    torch.Tensor: The result tensor of shape (batch, z, d) after applying the mask and computing the mean.
    """
    if not mask.dtype == torch.bool:
        raise TypeError("Mask must be a boolean tensor.")

    # Ensure the mask is broadcastable to the shape of x
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    masked_x = x * mask

    # Compute the sum and the count of valid (unmasked) elements along the x_size dimension
    sum_x = masked_x.sum(dim=1)
    count_x = mask.sum(dim=1)

    # Avoid division by zero
    count_x[count_x == 0] = 1

    # Compute the mean
    mean_x = sum_x / count_x

    return mean_x