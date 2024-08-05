import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            dim: int = 64,
            nhead: int = 8,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 2,
            dim_feedforward: int = 512,
            dropout: float = 0.01,
            activation: str = 'gelu',
            in_dim: int = 2,
            out_dim: int = None,
    ):

        super().__init__()

        self.in_projector = nn.Linear(in_dim, dim)

        self.dim = dim

        self.transformer = nn.Transformer(
            dim, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        if out_dim:
            self.out_projector = nn.Linear(dim, out_dim)
        else:
            self.out_projector = None

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, **kwargs):
        src = self.in_projector(src.transpose(1, 2)).transpose(0, 1)

        res = self.transformer(
            src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, **kwargs
        )

        if self.out_projector:
            res = self.out_projector(res).squeeze(-1)

        return res.squeeze(0)
