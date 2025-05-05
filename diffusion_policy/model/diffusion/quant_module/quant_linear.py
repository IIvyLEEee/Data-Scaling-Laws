import torch
import torch.nn as nn
import torch.nn.functional as F

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def int8_quantizer(x: torch.Tensor, delta: torch.Tensor):
    return torch.clamp(x / delta, -127, 127)

def int4_quantizer(x: torch.Tensor, delta: torch.Tensor):
    pass

class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor,
        weight: torch.Tensor,
        input_delta: torch.Tensor,
        weight_delta: torch.Tensor, 
        bias=None
        ):
        quant_input = int8_quantizer(input, input_delta)
        quant_weight = int8_quantizer(weight, weight_delta)
        output = quant_input @ quant_weight.transpose(0, 1)
        scaling_factor = input_delta * (2.0 ** (-weight_delta + 1)) 
        scaling_factor = scaling_factor.transpose(0, 1)
        dequant_output = output * scaling_factor
        
        ctx.save_for_backward(
            input, weight, input_delta, weight_delta, bias
        )
        return dequant_output

    @staticmethod
    def backward(ctx, dequant_grad_output: torch.Tensor):
        input, weight, input_delta, weight_delta, _ = (
            ctx.saved_tensors
        )
        dequant_grad_input = dequant_grad_weight = None
        dequant_grad_input = dequant_grad_output @ weight
        dequant_grad_weight = dequant_grad_output.transpose(-2, -1) @ input
        return dequant_grad_input, dequant_grad_weight, None, None, None, None

class QuantLinear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, bias: bool = False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.empty((out_feature, in_feature)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feature))
        else:
            self.register_parameter("bias", None)

        self.weight_n_bits = 8
        self.weight_n_levels = 2 ** (self.weight_n_bits - 1) - 1
        self.weight_delta = nn.Parameter(torch.empty((out_feature)), requires_grad=False)
        self.input_n_bits = 8
        self.input_n_levels = 2 ** (self.input_n_bits - 1) - 1
        self.input_delta = nn.Parameter(torch.full((1,), 0.0), requires_grad=False)
        self.calibration = False
        self.cali_p = 0.05
        self.init = True

        # 注册scale
        # self.register_buffer('input_delta', torch.tensor(0.))
        # self.register_buffer('weight_delta', torch.tensor(0.))
        # self.register_buffer('output_delta', torch.tensor(0.))
        # self.register_buffer('input_delta', None)
        # self.register_buffer('weight_delta', None)
        # self.register_buffer('output_delta', None)

        self.init = True
    
    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)
            
    def int8_init_scale(self, x: torch.Tensor):
        delta = None

        x_min = min(x.data.min().item(), 0)
        x_max = max(x.data.max().item(), 0)
        x_absmax = max(abs(x_min), x_max)
        delta = x_absmax / self.input_n_levels

        delta = torch.tensor(delta).type_as(x)
        return delta

    def forward(self, input: torch.Tensor, init=False):
        if self.init is True and self.input_delta.data == 0:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            self.weight_delta.data = self.int8_init_scale(weight_).requires_grad_(False) * 2
            self.input_delta.data = self.int8_init_scale(input_).requires_grad_(False) * 2

            self.init = False

        return LinearFunc.apply(
            input,
            self.weight,
            self.input_delta,
            self.weight_delta,
            self.bias,
        )

if __name__ == "__main__":
    in_features = 5
    batch_size = 10
    out_features = 5
    x = torch.randn(batch_size, in_features, requires_grad=True)  # .type(torch.int8)
    y = x.clone().detach().requires_grad_(True)
    w = torch.randn(out_features, in_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True).to(torch.bfloat16)

    layer = nn.Linear(in_features, out_features, True)
    layer.weight = nn.Parameter(w)
    # layer.bias = nn.Parameter(b).to(torch.bfloat16)
    out = layer(x)
    outy = y.matmul(w.transpose(0, 1))  # + b  # .type(torch.float32)

    # print("outy: ", outy)

    dout = torch.randn(batch_size, out_features) / 10  # .type(torch.float32)

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    # print("out: ", out)
    print("dx: ", x.grad)
    print("dy: ", y.grad)
    print(torch.min(x.grad / y.grad).item())
    print(torch.max(x.grad / y.grad).item())
