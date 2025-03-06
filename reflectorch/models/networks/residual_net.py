import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from reflectorch.models.activations import activation_by_name

class ResidualMLP(nn.Module):
    """Multilayer perceptron with residual blocks (BN-Act-Linear-BN-Act-Linear)"""

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            dim_condition: int = 0,
            layer_width: int = 512,
            num_blocks: int = 4,
            repeats_per_block: int = 2,
            activation: str = 'relu',
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            dropout_rate: float = 0.0,
            residual: bool = True,
            adaptive_activation: bool = False,
            conditioning: str = 'glu',
            concat_condition_first_layer: bool = True,
            film_with_tanh: bool = False,
    ):
        super().__init__()

        self.concat_condition_first_layer = concat_condition_first_layer

        dim_first_layer = dim_in + dim_condition if concat_condition_first_layer else dim_in
        self.first_layer = nn.Linear(dim_first_layer, layer_width)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    layer_width=layer_width,
                    dim_condition=dim_condition,
                    repeats_per_block=repeats_per_block,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                    residual=residual,
                    adaptive_activation=adaptive_activation,
                    conditioning = conditioning,
                    film_with_tanh = film_with_tanh,
                )
                for _ in range(num_blocks)
            ]
        )

        self.last_layer = nn.Linear(layer_width, dim_out)

    def forward(self, x, condition=None):
        if self.concat_condition_first_layer and condition is not None:
            x = self.first_layer(torch.cat([x, condition], dim=-1))
        else:
            x = self.first_layer(x)

        for block in self.blocks:
            x = block(x, condition=condition)

        x = self.last_layer(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block (BN-Act-Linear-BN-Act-Linear)"""

    def __init__(
            self,
            layer_width: int,
            dim_condition: int = 0,
            repeats_per_block: int = 2,
            activation: str = 'relu',
            use_batch_norm: bool = False,
            use_layer_norm: bool = False,
            dropout_rate: float = 0.0,
            residual: bool = True,
            adaptive_activation: bool = False,
            conditioning: str = 'glu',
            film_with_tanh: bool = False,
    ):
        super().__init__()
         
        self.residual = residual
        self.repeats_per_block = repeats_per_block
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.adaptive_activation = adaptive_activation
        self.conditioning = conditioning
        self.film_with_tanh = film_with_tanh

        if not adaptive_activation:
            self.activation = activation_by_name(activation)()
        else:
            self.activation_layers = nn.ModuleList(
                [activation_by_name(activation)() for _ in range(repeats_per_block)]
            )

        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(layer_width, eps=1e-3) for _ in range(repeats_per_block)]
             )
        elif use_layer_norm:
            self.layer_norm_layers = nn.ModuleList(
                [nn.LayerNorm(layer_width) for _ in range(repeats_per_block)]
             )

        if dim_condition:
            if conditioning == 'glu':
                self.condition_layer = nn.Linear(dim_condition, layer_width)
            elif conditioning == 'film':
                self.condition_layer = nn.Linear(dim_condition, 2*layer_width)

        self.linear_layers = nn.ModuleList(
            [nn.Linear(layer_width, layer_width) for _ in range(repeats_per_block)]
        )

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, condition=None):
        x0 = x
         
        for i in range(self.repeats_per_block):
            if self.use_batch_norm:
                x = self.batch_norm_layers[i](x)
            elif self.use_layer_norm:
                x = self.layer_norm_layers[i](x)

            if not self.adaptive_activation:
                x = self.activation(x)
            else:
                x = self.activation_layers[i](x)

            if self.dropout_rate > 0 and i == self.repeats_per_block - 1:
                x = self.dropout(x)

            x = self.linear_layers[i](x)
        
        if condition is not None:
            if self.conditioning == 'glu':
                x = F.glu(torch.cat((x, self.condition_layer(condition)), dim=-1), dim=-1)
            elif self.conditioning == 'film':
                gamma, beta = torch.chunk(self.condition_layer(condition), chunks=2, dim=-1)
                if self.film_with_tanh:
                    tanh = nn.Tanh()
                    gamma, beta = tanh(gamma), tanh(beta)
                x = x * gamma + beta

        return x0 + x if self.residual else x