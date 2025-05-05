import torch
import torch.nn as nn
import torch.nn.functional as F

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def int8_quantizer(x: torch.Tensor, delta: torch.Tensor):
    return torch.clamp(x / delta, -127, 127)

class Conv1dFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor,
        weight: torch.Tensor,
        input_delta: torch.Tensor,
        weight_delta: torch.Tensor,
        bias=None, 
        stride=1, 
        padding=0
        ):
        quant_input = int8_quantizer(input, input_delta)
        quant_weight = int8_quantizer(weight, weight_delta)

        output = F.conv1d(
            quant_input, 
            quant_weight, 
            bias=None, 
            stride=stride, 
            padding=padding, 
        )

        scaling_factor = input_delta * (2.0 ** (-weight_delta + 1)).view(-1, 1, 1)
        dequant_output = output * scaling_factor

        ctx.save_for_backward(
            input, weight, input_delta, weight_delta, bias,
        )
        ctx.stride = stride
        ctx.padding = padding

        return dequant_output

    @staticmethod
    def backward(ctx, dequant_grad_output: torch.Tensor):
        input, weight, input_delta, weight_delta, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        dequant_grad_input = dequant_grad_weight = None

        dequant_grad_input = F.conv_transpose1d(dequant_grad_output, weight, stride=stride, padding=padding)
        dequant_grad_weight = F.conv1d(input.transpose(0, 1), dequant_grad_output.transpose(0, 1), stride=stride, padding=padding).transpose(0, 1)

        return dequant_grad_input, dequant_grad_weight, None, None, None, None, None

class QuantConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias: bool =False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.weight_delta = nn.Parameter(torch.full((out_channels,), 0.0), requires_grad=False)
        self.input_delta = nn.Parameter(torch.full((1,), 0.0), requires_grad=False)
        self.calibration = False
        self.cali_p = 0.05
        self.init = True

    def int8_init_scale(self, x: torch.Tensor, n_levels=127):
        x_min = min(x.min().item(), 0)
        x_max = max(x.max().item(), 0)
        x_absmax = max(abs(x_min), x_max)
        delta = x_absmax / n_levels
        return torch.tensor(delta).type_as(x)

    def forward(self, input: torch.Tensor):
        if self.init and self.input_delta.data == 0:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            self.input_delta.data = self.int8_init_scale(input_).requires_grad_(False) * 2
            self.weight_delta.data = self.int8_init_scale(weight_).requires_grad_(False) * 2
            
            self.init = False

        return Conv1dFunc.apply(
            input, self.weight, self.input_delta, self.weight_delta, self.bias,
            self.stride, self.padding
        )

# 测试代码
if __name__ == "__main__":
    torch.manual_seed(0)

    # 参数
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    stride = 1
    padding = 1

    # 输入
    x = torch.randn(2, in_channels, 10, requires_grad=True)
    y = x.clone().detach().requires_grad_(True)

    # 创建标准 Conv1d 和自定义 QuantConv1d
    conv_ref = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    quant_conv = QuantConv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    # 权重同步
    with torch.no_grad():
        quant_conv.weight.copy_(conv_ref.weight)

    # 前向
    out_ref = conv_ref(x)
    out_quant = quant_conv(y)

    # 创建相同的梯度进行反向传播
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_quant.backward(grad)

    # 输出结果
    print("dx (nn.Conv1d input grad):", x.grad)
    print("dy (QuantConv1d input grad):", y.grad)

    # 避免除以0
    ratio = x.grad / (y.grad + 1e-6)

    print("Gradient ratio min:", torch.min(ratio).item())
    print("Gradient ratio max:", torch.max(ratio).item())
    print("Gradient MSE:", ((x.grad - y.grad) ** 2).mean().item())