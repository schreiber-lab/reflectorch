import math
from typing import Optional
import torch
from torch import Tensor, nn

from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.fno import FnoEncoder
from reflectorch.models.encoders.integral_kernel_embedding import IntegralConvEmbedding
from reflectorch.models.networks.nf.nf import get_residual_transform_net_fn, get_rq_nsf_c_flow

class NFNetwork(nn.Module):
    def __init__(self,
                 embedding_net_type: str,
                 embedding_net_kwargs: dict,
                 transform_net_kwargs: dict,
                 flow_kwargs: dict,
                 pretrained_embedding_net: str = None,
                 dim_theta: int = 8,
                 dim_conditioning_params: int = 0,
                 prior_bounds_input : bool = True,
                 ):
        super().__init__()

        self.dim_theta = dim_theta
        self.dim_prior_bounds = 2 * dim_theta if prior_bounds_input else 0
        self.dim_conditioning_params = dim_conditioning_params
        self.dim_embedding = embedding_net_kwargs['dim_embedding']
        self.dim_condition = self.dim_embedding + self.dim_prior_bounds + self.dim_conditioning_params
        self.prior_bounds_input = prior_bounds_input

        if embedding_net_type == 'conv':
            self.embedding_net = ConvEncoder(**embedding_net_kwargs)
        elif embedding_net_type == 'fno':
            self.embedding_net = FnoEncoder(**embedding_net_kwargs)
        elif embedding_net_type == 'integral_conv':
            self.embedding_net = IntegralConvEmbedding(**embedding_net_kwargs)
        elif embedding_net_type == 'no_embedding_net':
            self.embedding_net = nn.Identity()
        else:
            raise ValueError(f"Unsupported embedding_net_type: {embedding_net_type}")


        transform_net_fn = get_residual_transform_net_fn(
            context_features=self.dim_condition,
            **transform_net_kwargs
        )

        self.flow = get_rq_nsf_c_flow(
            features=self.dim_theta,
            transform_net_fn=transform_net_fn,
            **flow_kwargs,
        )

        if pretrained_embedding_net:
            self.embedding_net.load_weights(pretrained_embedding_net)


    def sample(self, num_samples, curves, bounds=None, q_values=None, sigmas=None, conditioning_params=None, key_padding_mask=None,
               unscaled_q_values=None, batch_size=None):
        """
        """

        if curves.dim() == 2:
            curves = curves.unsqueeze(1)

        additional_channels = []
        if q_values is not None and not isinstance(self.embedding_net, IntegralConvEmbedding):
            additional_channels.append(q_values.unsqueeze(1))
        if sigmas is not None:
            additional_channels.append(sigmas.unsqueeze(1))

        if additional_channels:
            curves = torch.cat([curves] + additional_channels, dim=1)  # [batch_size, n_channels, n_points]

        if isinstance(self.embedding_net, IntegralConvEmbedding):
            x = self.embedding_net(q=unscaled_q_values.float(), y=curves.permute(0, 2, 1), drop_mask=key_padding_mask)
        else:
            x = self.embedding_net(curves)


        condition = torch.cat([x] + ([bounds] if bounds is not None else []) + ([conditioning_params] if conditioning_params is not None else []), dim=-1)
        samples = self.flow.sample(num_samples=num_samples, context=condition, batch_size=batch_size)
        
        return samples
    
    def sample_and_log_prob(self, num_samples, curves, bounds=None, q_values=None, sigmas=None, conditioning_params=None, key_padding_mask=None, unscaled_q_values=None):

        if curves.dim() == 2:
            curves = curves.unsqueeze(1)

        additional_channels = []
        if q_values is not None and not isinstance(self.embedding_net, IntegralConvEmbedding):
            additional_channels.append(q_values.unsqueeze(1))
        if sigmas is not None:
            additional_channels.append(sigmas.unsqueeze(1))

        if additional_channels:
            curves = torch.cat([curves] + additional_channels, dim=1)  # [batch_size, n_channels, n_points]

        if isinstance(self.embedding_net, IntegralConvEmbedding):
            x = self.embedding_net(q=unscaled_q_values.float(), y=curves.permute(0, 2, 1), drop_mask=key_padding_mask)
        else:
            x = self.embedding_net(curves)


        condition = torch.cat([x] + ([bounds] if bounds is not None else []) + ([conditioning_params] if conditioning_params is not None else []), dim=-1)
        samples = self.flow.sample_and_log_prob(num_samples=num_samples, context=condition)
        
        return samples
    
    def log_prob(self, inputs, curves, bounds=None, q_values=None, sigmas=None, conditioning_params=None, key_padding_mask=None, unscaled_q_values=None):
        
        if curves.dim() == 2:
            curves = curves.unsqueeze(1)

        additional_channels = []
        if q_values is not None and not isinstance(self.embedding_net, IntegralConvEmbedding):
            additional_channels.append(q_values.unsqueeze(1))
        if sigmas is not None:
            additional_channels.append(sigmas.unsqueeze(1))

        if additional_channels:
            curves = torch.cat([curves] + additional_channels, dim=1)  # [batch_size, n_channels, n_points]

        if isinstance(self.embedding_net, IntegralConvEmbedding):
            x = self.embedding_net(q=unscaled_q_values.float(), y=curves.permute(0, 2, 1), drop_mask=key_padding_mask)
        else:
            x = self.embedding_net(curves)

        condition = torch.cat([x] + ([bounds] if bounds is not None else []) + ([conditioning_params] if conditioning_params is not None else []), dim=-1)
        log_prob = self.flow.log_prob(inputs=inputs, context=condition)
        
        return log_prob
    
if __name__ == "__main__":
    
    nf_network = NFNetwork(
        dim_theta = 10,
        dim_conditioning_params=0,
        embedding_net_type='conv',
        embedding_net_kwargs={
            'in_channels': 1,
            'hidden_channels': [32, 64, 128, 256, 512],
            'kernel_size': 3,
            'dim_embedding': 128,
            'dim_avpool': 1,
            'use_batch_norm': True,
            'activation': 'gelu',
        },
        transform_net_kwargs={
            'hidden_features': 64,
            'activation': "lrelu",
            'use_batch_norm': True,
            'num_blocks': 3,
        },
        flow_kwargs={
            'num_layers': 10,
            'tail_bound': 10.0,
            'tails': "linear",
            'num_bins': 8,
            'use_batch_norm_transform': True,
            'use_lu': False,
        },
    )

    print(nf_network)