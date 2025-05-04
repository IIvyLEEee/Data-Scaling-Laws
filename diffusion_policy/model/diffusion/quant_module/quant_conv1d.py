import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def int_quantizer(x: torch.Tensor, delta: torch.Tensor, n_bits: int):
    n_levels = 2 ** (n_bits - 1) - 1
    x_int = round_ste(x / delta)
    x_quant = torch.clamp(x_int, -n_levels, n_levels)
    return x_quant

def int_dequantizer(x_quant: torch.Tensor, delta: torch.Tensor):
    return x_quant * delta

class Conv1dFunc(torch.autograd.Function):
    """自定义量化Conv1d的autograd Function"""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_delta: torch.Tensor,
        weight_delta: torch.Tensor,
        output_delta: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        input_bits=8,
        weight_bits=4
    ):
        # 输入量化
        quant_input = int_quantizer(input, input_delta, input_bits)
        
        # 权重量化
        quant_weight = int_quantizer(weight, weight_delta, weight_bits)
        
        # 执行量化后的卷积
        output = F.conv1d(
            quant_input, 
            quant_weight, 
            bias=bias, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups
        )
        
        # 计算输出缩放因子
        scaling_factor = (input_delta * weight_delta) / output_delta
        
        # 输出量化和反量化
        quant_output = output * scaling_factor
        dequant_output = quant_output * output_delta
        
        # 保存用于反向传播的变量
        ctx.save_for_backward(
            input, weight, input_delta, weight_delta, output_delta, 
            torch.tensor([padding]), torch.tensor([stride]), 
            torch.tensor([dilation]), torch.tensor([groups])
        )
        ctx.input_bits = input_bits
        ctx.weight_bits = weight_bits
        
        return dequant_output

    @staticmethod
    def backward(ctx, dequant_grad_output):
        # 获取保存的变量
        input, weight, input_delta, weight_delta, output_delta, \
        padding, stride, dilation, groups = ctx.saved_tensors
        
        # 计算输入的梯度
        dequant_grad_input = F.conv_transpose1d(
            dequant_grad_output, 
            weight, 
            stride=stride.item(), 
            padding=padding.item(), 
            dilation=dilation.item(), 
            groups=groups.item()
        )
        
        # 计算权重的梯度
        dequant_grad_weight = F.conv1d(
            input.transpose(0, 1), 
            dequant_grad_output.transpose(0, 1), 
            padding=padding.item(), 
            stride=dilation.item(), 
            dilation=stride.item(), 
            groups=groups.item()
        ).transpose(0, 1)
        
        # return dequant_grad_input, dequant_grad_weight, None, None, None, None, None, None, None, None, None
        return (
            dequant_grad_input,  # 对应 input
            dequant_grad_weight,  # 对应 weight
            None,  # 对应 input_delta
            None,  # 对应 weight_delta
            None,  # 对应 output_delta
            None,  # 对应 bias
            None,  # 对应 stride
            None,  # 对应 padding
            None,  # 对应 dilation
            None,  # 对应 groups
            None,  # 对应 input_bits
            None   # 对应 weight_bits
        )

class QuantConv1d(nn.Module):
    """量化Conv1d层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        input_bits: int = 8,
        weight_bits: int = 16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 初始化权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        
        # 初始化偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 量化参数
        self.input_bits = input_bits
        self.weight_bits = weight_bits

        # 注册scale
        # self.register_buffer('input_delta', torch.tensor(0.))
        # self.register_buffer('weight_delta', torch.tensor(0.))
        # self.register_buffer('output_delta', torch.tensor(0.))
        self.register_buffer('input_delta', None)
        self.register_buffer('weight_delta', None)
        self.register_buffer('output_delta', None)


        self.init = True
        
        # 初始化权重和偏置
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def init_scale(self, x: torch.Tensor, n_bits: int):
        """初始化比例因子"""
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
            
            # 计算输出用于初始化输出比例因子
            output_ = F.conv1d(
                input_, 
                weight_, 
                bias=self.bias, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=self.dilation, 
                groups=self.groups
            )
            
            # 初始化各比例因子
            self.weight_delta = self.init_scale(weight_, self.weight_bits).requires_grad_(False)
            self.input_delta = self.init_scale(input_, self.input_bits).requires_grad_(False)
            self.output_delta = self.init_scale(output_, self.input_bits).requires_grad_(False)
            self.init = False
        
        # 调用自定义的autograd Function
        return Conv1dFunc.apply(
            input,
            self.weight,
            self.input_delta,
            self.weight_delta,
            self.output_delta,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.input_bits,
            self.weight_bits
        )

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 4
    in_channels = 3
    out_channels = 6
    length = 32
    kernel_size = 3
    input_bits = 8
    weight_bits = 4
    
    # 创建输入和普通Conv1d层
    x = torch.randn(batch_size, in_channels, length, requires_grad=True)
    y = x.clone().detach().requires_grad_(True)
    
    # 创建量化Conv1d层和普通Conv1d层
    quant_conv = QuantConv1d(
        in_channels, out_channels, kernel_size, 
        input_bits=input_bits, weight_bits=weight_bits
    )
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
    
    # 复制权重以确保公平比较
    with torch.no_grad():
        conv.weight.copy_(quant_conv.weight)
    
    # 前向传播
    quant_out = quant_conv(x)
    out = conv(y)
    
    # 计算梯度
    dout = torch.randn_like(quant_out) / 10
    
    # 量化层的反向传播
    fake_loss = (quant_out * dout).sum()
    fake_loss.backward()
    
    # 普通层的反向传播
    loss = (out * dout).sum()
    loss.backward()

    # # 创建一个测试输入
    # x1 = torch.randn(1, 3, 32)  # FP32 输入

    # # 创建量化层
    # quant_conv = QuantConv1d(3, 6, 3, input_bits=4, weight_bits=4)

    # # 前向传播
    # with torch.no_grad():
    #     output = quant_conv(x1)
        
    # # 检查输入是否被量化
    # print("原始输入值范围:", x1.min(), x1.max())
    # print("量化后的输入值范围:", (x1/quant_conv.input_delta).round().clamp(-7, 7).min(), 
    #                             (x1/quant_conv.input_delta).round().clamp(-7, 7).max())
    
    # 比较梯度
    print("输入梯度比较:")
    print("量化层梯度:", x.grad[0, 0, :5])
    print("普通层梯度:", y.grad[0, 0, :5])
    print("梯度比例范围:", torch.min(x.grad / y.grad), "~", torch.max(x.grad / y.grad))
    
    print("\n权重梯度比较:")
    print("量化层梯度:", quant_conv.weight.grad[0, 0, :5])
    print("普通层梯度:", conv.weight.grad[0, 0, :5])
    print("梯度比例范围:", torch.min(quant_conv.weight.grad / conv.weight.grad), 
          "~", torch.max(quant_conv.weight.grad / conv.weight.grad))