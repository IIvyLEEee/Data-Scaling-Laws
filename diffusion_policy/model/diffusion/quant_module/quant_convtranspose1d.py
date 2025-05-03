import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def round_ste(x: torch.Tensor):
    """Straight-Through Estimator的舍入操作"""
    return (x.round() - x).detach() + x

def int_quantizer(x: torch.Tensor, delta: torch.Tensor, n_bits: int):
    """通用整数量化函数"""
    n_levels = 2 ** (n_bits - 1) - 1
    x_int = round_ste(x / delta)
    x_quant = torch.clamp(x_int, -n_levels, n_levels)
    return x_quant

def int_dequantizer(x_quant: torch.Tensor, delta: torch.Tensor):
    """反量化函数"""
    return x_quant * delta

class QuantConvTranspose1d(nn.Module):
    """量化ConvTranspose1d层（自动求导版本）"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        input_bits: int = 8,
        weight_bits: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.input_bits = input_bits
        self.weight_bits = weight_bits

        # 注册scale
        self.register_buffer('input_delta', torch.tensor(0.))
        self.register_buffer('weight_delta', torch.tensor(0.))
        self.register_buffer('output_delta', torch.tensor(0.))

        # 初始化
        self.init = True

        # 初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def init_scale(self, x: torch.Tensor, n_bits: int):
        x_min = x.min().item()
        x_max = x.max().item()
        x_absmax = max(abs(x_min), x_max)
        n_levels = 2 ** (n_bits - 1) - 1
        delta = x_absmax / n_levels
        return torch.tensor(delta).to(x.device)

    def forward(self, input: torch.Tensor):
        # 初始化比例因子
        if self.init and self.weight_delta is None:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            output_ = F.conv_transpose1d(
                input_,
                weight_,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups
            )
            self.weight_delta = self.init_scale(weight_, self.weight_bits).requires_grad_(False)
            self.input_delta = self.init_scale(input_, self.input_bits).requires_grad_(False)
            self.output_delta = self.init_scale(output_, self.input_bits).requires_grad_(False)
            self.init = False
        # 量化输入和权重
        quant_input = int_quantizer(input, self.input_delta, self.input_bits)
        quant_weight = int_quantizer(self.weight, self.weight_delta, self.weight_bits)
        # 反卷积
        output = F.conv_transpose1d(
            quant_input, quant_weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, dilation=self.dilation, groups=self.groups
        )
        # 反量化
        scaling_factor = (self.input_delta * self.weight_delta) / self.output_delta
        quant_output = output * scaling_factor
        dequant_output = quant_output * self.output_delta
        return dequant_output

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 4
    in_channels = 6
    out_channels = 3
    length = 32
    kernel_size = 3
    input_bits = 8
    weight_bits = 4
    x = torch.randn(batch_size, in_channels, length, requires_grad=True)
    y = x.clone().detach().requires_grad_(True)
    quant_conv_trans = QuantConvTranspose1d(
        in_channels, out_channels, kernel_size, 
        input_bits=input_bits, weight_bits=weight_bits
    )
    conv_trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, bias=False)
    
    with torch.no_grad():
        conv_trans.weight.copy_(quant_conv_trans.weight)
    
    quant_out = quant_conv_trans(x)
    out = conv_trans(y)
    dout = torch.randn_like(quant_out) / 10
    fake_loss = (quant_out * dout).sum()
    fake_loss.backward()
    loss = (out * dout).sum()
    loss.backward()
    
    print("输入梯度比较:")
    print("量化层梯度:", x.grad[0, 0, :5])
    print("普通层梯度:", y.grad[0, 0, :5])
    print("梯度比例范围:", torch.min(x.grad / y.grad), "~", torch.max(x.grad / y.grad))
    print("\n权重梯度比较:")
    print("量化层梯度:", quant_conv_trans.weight.grad[0, 0, :5])
    print("普通层梯度:", conv_trans.weight.grad[0, 0, :5])
    print("梯度比例范围:", torch.min(quant_conv_trans.weight.grad / conv_trans.weight.grad), 
          "~", torch.max(quant_conv_trans.weight.grad / conv_trans.weight.grad))