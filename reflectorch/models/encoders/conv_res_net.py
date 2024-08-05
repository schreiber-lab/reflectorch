 

from torch import nn
from torch.nn import functional as F
from torch.nn import init

__all__ = [
    'ConvResidualNet1D',
]


class ConvResidualBlock1D(nn.Module):
    def __init__(
            self,
            channels,
            activation=F.gelu,
            dropout_probability=0.0,
            use_batch_norm=False,
            zero_initialization=True,
            kernel_size: int = 3,
            dilation: int = 1,
            padding: int = 1,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
             for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)

        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)

        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)

        return inputs + temps


class ConvResidualNet1D(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 64,
            hidden_channels: int = 128,
            num_blocks=5,
            activation=F.gelu,
            dropout_probability=0.0,
            use_batch_norm=True,
            kernel_size: int = 3,
            dilation: int = 1,
            padding: int = 1,
            avpool: int = 8,

    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.initial_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock1D(
                    channels=hidden_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for _ in range(num_blocks)
            ]
        )

        self.avpool = nn.AdaptiveAvgPool1d(avpool)

        self.final_layer = nn.Linear(
            hidden_channels * avpool, out_channels
        )

    def forward(self, x):
        temps = self.initial_layer(x.unsqueeze(1))

        for block in self.blocks:
            temps = block(temps)

        temps = self.avpool(temps).view(temps.size(0), -1)
        outputs = self.final_layer(temps)

        return outputs
