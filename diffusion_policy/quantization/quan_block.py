import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from diffusion_policy.quantization.quan_layer import (
    StraightThrough,
    Activation_Quantizer,
    Weight_Quantizer,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """

    def __init__(self):
        super().__init__()

        self.act_quantizer = Activation_Quantizer()
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class QuantPosEmbBlock(BaseQuantBlock):
    def __init__(self, pos_emb: SinusoidalPosEmb):
        super().__init__()
        self.dim = pos_emb.dim
        self.emb_quantizer = Weight_Quantizer()

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        if self.use_act_quant:
            x = self.act_quantizer(x)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if self.use_act_quant:
            emb = self.emb_quantizer(emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant
        self.use_weight_quant = weight_quant


def get_specials():
    specials = {
        # SinusoidalPosEmb: QuantPosEmbBlock,
        # nn.MultiheadAttention: QuantMultihead
    }

    return specials
