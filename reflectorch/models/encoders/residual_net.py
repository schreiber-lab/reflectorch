import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from reflectorch.models.utils import activation_by_name

class ResidualMLP(nn.Module):
    """Multilayer perceptron with residual blocks (BN-Act-Linear-BN-Act-Linear)"""

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            dim_condition: int = None,
            layer_width: int = 512,
            num_blocks: int = 4,
            repeats_per_block: int = 2,
            activation: str = 'relu',
            use_batch_norm: bool = True,
            dropout_rate: float = 0.0,
            residual: bool = True,
            adaptive_activation: bool = False,
    ):
        super().__init__()

        dim_first_layer = dim_in + dim_condition if dim_condition is not None else dim_in 
        self.first_layer = nn.Linear(dim_first_layer, layer_width)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    layer_width=layer_width,
                    dim_condition=dim_condition,
                    repeats_per_block=repeats_per_block,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                    residual=residual,
                    adaptive_activation=adaptive_activation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.last_layer = nn.Linear(layer_width, dim_out)

    def forward(self, x, condition=None):
        if condition is None:
            x = self.first_layer(x)
        else:
            x = self.first_layer(torch.cat((x, condition), dim=1))

        for block in self.blocks:
            x = block(x, condition=condition)
        x = self.last_layer(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block (BN-Act-Linear-BN-Act-Linear)"""

    def __init__(
            self,
            layer_width: int,
            dim_condition: int = None,
            repeats_per_block: int = 2,
            activation: str = 'relu',
            use_batch_norm: bool = False,
            dropout_rate: float = 0.0,
            residual: bool = True,
            adaptive_activation: bool = False,
    ):
        super().__init__()
         
        self.residual = residual
        self.repeats_per_block = repeats_per_block
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.adaptive_activation = adaptive_activation

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

        if dim_condition is not None:
            self.condition_layer = nn.Linear(dim_condition, layer_width)

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
            if not self.adaptive_activation:
                x = self.activation(x)
            else:
                x = self.activation_layers[i](x)
            if self.dropout_rate > 0 and i == self.repeats_per_block - 1:
                x = self.dropout(x)
            x = self.linear_layers[i](x)
        
        if condition is not None:
            x = F.glu(torch.cat((x, self.condition_layer(condition)), dim=1), dim=1)

        return x0 + x if self.residual else x

    
class ResidualMLP_FiLM(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
            self,
            in_features,
            prior_in_features,
            out_features,
            hidden_features,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features

        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock_FiLM(
                    features=hidden_features,
                    prior_in_features=prior_in_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, prior_bounds):
        temps = self.initial_layer(inputs)

        for block in self.blocks:
            temps = block(temps, prior_bounds)
        outputs = self.final_layer(temps)
        return outputs


class ResidualBlock_FiLM(nn.Module):
    def __init__(
            self,
            features,
            prior_in_features,
            activation,
            dropout_probability=0.0,
            use_batch_norm=False,
            zero_initialization=True,
    ):
        super().__init__()
        
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
            
        
        self.film_layer_gamma = nn.Linear(prior_in_features, features)
        self.film_layer_beta = nn.Linear(prior_in_features, features)
            
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, prior_bounds):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        
        temps = self.film_layer_gamma(prior_bounds)*temps + self.film_layer_beta(prior_bounds)
        
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)

        return inputs + temps