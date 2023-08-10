# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from reflectorch.models.encoders.conv_encoder import (
    ConvEncoder,
    ConvDecoder,
    Autoencoder,
)
from reflectorch.models.encoders.fno import FNO_Enc
from reflectorch.models.encoders.transformers import TransformerEncoder
from reflectorch.models.encoders.conv_res_net import ConvResidualNet1D
from reflectorch.models.encoders.conv_res_net import ConvResidualNet1D
from reflectorch.models.encoders.conv_fc_encoder import (
    ConvFCEncoder,
    SubPriorConvFCEncoder,
    SubPriorConvFCEncoder_V2,
    SubPriorConvFCEncoder_FiLM,
    SubPrior_FNO_MLP,
    SeqConvFCEncoder,
    ResidualNet_FiLM,
)
from reflectorch.models.encoders.residual_net import ResidualNet


__all__ = [
    "TransformerEncoder",
    "ConvEncoder",
    "ConvDecoder",
    "Autoencoder",
    "ConvResidualNet1D",
    "FNO_Enc",
    "ResidualNet",
    "ResidualNet_FiLM",
    "ConvFCEncoder",
    "SubPriorConvFCEncoder",
    "SubPriorConvFCEncoder_V2",
    "SubPriorConvFCEncoder_FiLM",
    "SubPrior_FNO_MLP",
    "SeqConvFCEncoder",
]
