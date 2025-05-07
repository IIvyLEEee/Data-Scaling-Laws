import logging
import math
import warnings
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class Weight_Quantizer(nn.Module):
    def __init__(self):
        super(Weight_Quantizer, self).__init__()
        self.delta = None
        self.inited = False

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            delta = self.init_quantization_scale(x, True)
            self.delta = torch.nn.Parameter(delta.float())
            self.inited = True

        # start quantization
        # self.delta = b
        # a = 2 ** (-b)
        # v = 2 ** (w_log_scales - 1)
        w_max = (2 - 2 ** (-1)) * 2 ** (2**2 - 1 - self.delta)
        w_min = -w_max
        x_R = torch.min(torch.max(x, w_min), w_max)
        w_log_scales = torch.clamp(
            (torch.floor(torch.log2(torch.abs(x_R) + 1e-5) + self.delta)).detach(), 1.0
        )
        x_scales = 2.0 ** (w_log_scales - 1 - self.delta)
        x_quant_m = (x / x_scales).round_()
        x_dequant = x_quant_m.mul_(x_scales)
        # print(x-x_dequant)
        # print(x_dequant/2.0**(- self.delta)/2)

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta = None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
        else:
            w_max = x.abs().max()
            delta = 2**2 - torch.log2(w_max) + math.log2(2 - 2 ** (-1)) - 1
            delta = delta.clone().detach().type_as(x)

        return delta


class Activation_Quantizer(nn.Module):
    def __init__(self):
        super(Activation_Quantizer, self).__init__()
        self.n_bits = 8
        self.n_levels = 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.running_stat = False
        self.x_min, self.x_max = None, None

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            delta, self.zero_point = self.init_quantization_scale(x)
            self.delta = torch.nn.Parameter(delta)
            # self.zero_point = torch.nn.Parameter(self.zero_point)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert self.inited

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)
        delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        delta = torch.clamp(delta, min=1e-8)
        self.delta = torch.nn.Parameter(delta)

    def init_quantization_scale(self, x: torch.Tensor):
        delta, zero_point = None, None
        self.x_min = x.data.min()
        self.x_max = x.data.max()

        x_min = min(x.min().item(), 0)
        x_max = max(x.max().item(), 0)
        # if 'scale' in self.scale_method:
        #     x_min = x_min * (self.n_bits + 2) / 8
        #     x_max = x_max * (self.n_bits + 2) / 8

        x_absmax = max(abs(x_min), x_max)
        delta = x_absmax / self.n_levels
        if delta < 1e-8:
            warnings.warn(
                "Quantization range close to zero: [{}, {}]".format(x_min, x_max)
            )
            delta = 1e-8

        zero_point = 0
        delta = torch.tensor(delta).type_as(x)
        return delta, zero_point


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d]):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # initialize quantizer
        self.weight_quantizer = Weight_Quantizer()
        self.act_quantizer = Activation_Quantizer()

        self.activation_function = StraightThrough()

    def forward(self, input: torch.Tensor):
        if self.use_act_quant:
            input = self.act_quantizer(input)
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat
