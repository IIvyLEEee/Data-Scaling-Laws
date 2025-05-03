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
    def __init__(self, in_features, out_features, bias=True, input_bits=8, weight_bits=4):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.input_bits = input_bits
        self.weight_bits = weight_bits

        # 注册scale
        self.register_buffer('input_delta', torch.tensor(0.))
        self.register_buffer('weight_delta', torch.tensor(0.))
        self.register_buffer('output_delta', torch.tensor(0.))

        self.init = True
        
        # 初始化权重和偏置
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def init_scale(self, x, n_bits):
        """
        根据输入张量x和bit数n_bits，计算量化比例因子scale。
        返回值为Tensor，shape与x的统计方式有关，通常为标量。
        """
        # 防止x为None
        if x is None:
            raise ValueError("init_scale: 输入x不能为None")
        # 计算scale，防止全零导致除零
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (2 ** n_bits - 1)
        # 防止scale为0
        if scale == 0:
            scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return scale
            
    def forward(self, input):
        # 初始化比例因子
        if self.init and self.weight_delta is None:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            output_ = F.linear(input_, weight_, self.bias)
            # print("weight_.shape: ", weight_.shape)
            # print("input_.shape: ", input_.shape)
            # print("output_.shape: ", output_.shape)
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