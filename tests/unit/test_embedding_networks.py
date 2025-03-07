import pytest
import torch
from reflectorch.models.encoders import ConvEncoder, FnoEncoder

@pytest.mark.parametrize("input_dim", [128, 140])
@pytest.mark.parametrize("in_channels", [1])
@pytest.mark.parametrize("hidden_channels", [(32, 64, 128, 256, 512), (16, 32, 64, 128, 256)])
@pytest.mark.parametrize("dim_embedding", [64, 80])
@pytest.mark.parametrize("dim_avpool", [1, 4])
@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("activation", ['relu', 'gelu'])
def test_cnn(input_dim, in_channels, hidden_channels, dim_embedding, dim_avpool, use_batch_norm, activation):
    embedding_net = ConvEncoder(
        in_channels=in_channels, 
        hidden_channels=hidden_channels,
        dim_embedding=dim_embedding,
        dim_avpool=dim_avpool,
        use_batch_norm=use_batch_norm,
        activation=activation,
        )
    
    batch_size = 4
    input_tensor = torch.randn((batch_size, in_channels, input_dim))

    assert embedding_net(input_tensor).shape == (batch_size, dim_embedding)

@pytest.mark.parametrize("input_dim", [128, 140])
@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("dim_embedding", [64, 80])
@pytest.mark.parametrize("modes", [16, 32])
@pytest.mark.parametrize("fusion_self_attention", [False, True])
def test_fno(input_dim, in_channels, dim_embedding, modes, fusion_self_attention):
    embedding_net = FnoEncoder(
        in_channels = in_channels,
        dim_embedding = dim_embedding,
        modes = modes,
        width_fno = 128,
        n_fno_blocks = 2,
        activation = 'gelu',
        fusion_self_attention=fusion_self_attention,
    )

    batch_size = 4
    input_tensor = torch.randn((batch_size, in_channels, input_dim))

    assert embedding_net(input_tensor).shape == (batch_size, dim_embedding)