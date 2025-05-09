import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import int4_kernel as ik
from diffusion_policy.quantization.quan_fwd import conv1d_int8_fwd, linear_int8_fwd

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
        self.n_bits = 8
        self.n_levels = 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, True)
            self.inited = True

        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
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
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(
                    x_clone[c], channel_wise=False
                )
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)

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
            # self.fwd_func = conv1d_int8_fwd
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            # self.fwd_func = linear_int8_fwd
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
        input = input.to(torch.bfloat16)
        weight = self.weight.to(torch.bfloat16)
        bias = self.bias

        if self.fwd_func == F.conv1d:
            self.fwd_func = conv1d_int8_fwd
        elif self.fwd_func == F.linear:
            self.fwd_func = linear_int8_fwd
        else:
            self.fwd_func = self.fwd_func

        print(self.fwd_kwargs)
        # import ipdb; ipdb.set_trace()
        out, scale_x, scale_w = self.fwd_func(x_bf16=input, weight_bf16=weight, use_act_quant=self.use_act_quant, \
                                             use_weight_quant=self.use_weight_quant, **self.fwd_kwargs)
        
        out = self.activation_function(out)

        scale_out = scale_x * scale_w
        out_dequant = out.to(torch.float32) * scale_out
    
        if bias is not None:
            # import ipdb; ipdb.set_trace()
            print(f"bias shape: {bias.shape}")
            print(f"out_dequant shape: {out_dequant.shape}")
            out_dequant = self.add_bias_broadcast(out_dequant, bias)
            # out_dequant = out_dequant + bias

        return out_dequant

    # def forward(self, input: torch.Tensor):
    #     if self.use_act_quant:
    #         input = self.act_quantizer(input)
    #     if self.use_weight_quant:
    #         weight = self.weight_quantizer(self.weight)
    #         bias = self.bias
    #     else:
    #         weight = self.org_weight
    #         bias = self.org_bias

    #     print(self.fwd_func)
    #     # import ipdb; ipdb.set_trace()
    #     out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
    #     out = self.activation_function(out)

        # return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat

    def add_bias_broadcast(self, out, bias):
        """
        自动将 bias reshape 成可广播到 out 的形状再相加。
        """
        if bias.ndim != 1:
            raise ValueError("bias 应为 1D tensor")

        # 在哪些维度能匹配 bias.shape？
        for dim in range(out.ndim):
            if out.shape[dim] == bias.shape[0]:
                # reshape bias 以匹配该维度，其它位置为 1
                shape = [1] * out.ndim
                shape[dim] = bias.shape[0]
                return out + bias.view(*shape)

        raise ValueError(f"找不到与 bias.shape {bias.shape} 匹配的 out.shape {out.shape} 维度")
