import torch
import torch.nn as nn
import torch.nn.functional as F

def round_ste(x: torch.Tensor):
    return torch.round(x.clamp(-128, 127))

def int_quantizer(x: torch.Tensor, delta: torch.Tensor, n_bits: int):
    return round_ste(x / delta) * delta

def int_dequantizer(x_quant: torch.Tensor, delta: torch.Tensor):
    return x_quant * delta

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.input_bits = 8
        self.weight_bits = 16
        self.init = True
        self.weight_delta = None
        self.input_delta = None
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def init_scale(self, x, n_bits):
        if self.init:
            self.weight_delta = torch.ones_like(self.weight)
            self.input_delta = torch.ones_like(x)
            self.init = False
        else:
            self.weight_delta = self.init_scale(self.weight, self.weight_bits).requires_grad_(False)
            self.input_delta = self.init_scale(x, self.input_bits).requires_grad_(False)
            self.output_delta = self.init_scale(self.weight @ x, self.input_bits).requires_grad_(False)
            self.init = False
            
    def forward(self, input):
        # 初始化比例因子
        if self.init and self.weight_delta is None:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            output_ = F.linear(input_, weight_, self.bias)
            self.weight_delta = self.init_scale(weight_, self.weight_bits).requires_grad_(False)
            self.input_delta = self.init_scale(input_, self.input_bits).requires_grad_(False)
            self.output_delta = self.init_scale(output_, self.input_bits).requires_grad_(False)
            self.init = False
        # 量化输入和权重
        quant_input = int_quantizer(input, self.input_delta, self.input_bits)
        quant_weight = int_quantizer(self.weight, self.weight_delta, self.weight_bits)
        # 线性变换
        output = F.linear(quant_input, quant_weight, self.bias)
        # 反量化
        scaling_factor = (self.input_delta * self.weight_delta) / self.output_delta
        quant_output = output * scaling_factor
        dequant_output = quant_output * self.output_delta
        return dequant_output
    
def test():
    quant_linear = QuantLinear(10, 10)
    input = torch.randn(10)
    output = quant_linear(input)
    print(output)

if __name__ == "__main__":
    test()