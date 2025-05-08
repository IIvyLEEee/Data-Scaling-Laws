import torch
import torch.nn as nn
import torch.nn.functional as F
import int4_kernel as ik
from torch.nn.modules.utils import _pair
# from diffusion_policy.quantization.quan_layer import Activation_Quantizer
# from diffusion_policy.quantization.quan_layer import Weight_Quantizer

"""
Notice:
    x and weight are expected to be in bf16 format.
"""

# aq = Activation_Quantizer()
# wq = Weight_Quantizer()
delta = 2.0

def conv1d_int8_fwd(x_bf16, weight_bf16, kernel_size=3, stride=1, padding=0, \
                    bias=None, dilation=1, groups=1, use_act_quant=True, use_weight_quant=True):
    # import ipdb; ipdb.set_trace()
    assert groups == 1, "Only groups=1 is supported in this implementation"

    print(f"conv1d_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    print(f"padding: {padding}")
    padding = padding[0]
    print(f"padding: {padding}")
    # stride = _pair(stride[0])
    stride = stride[0]
    print(f"stride: {stride}")
    kernel_size = weight_bf16.shape[2]
    print(f"kernel_size: {kernel_size}")
    # kernel_size = _pair(kernel_size)
    # print(f"kernel_size: {kernel_size}")

    x_padded = F.pad(x_bf16.unsqueeze(-1), pad=(0, 0, padding, padding))
    print(f"x_padded shape: {x_padded.shape}")  # [B, C_in, W_in+2*padding, 1]
    x_unf = F.unfold(x_padded, kernel_size=(kernel_size, 1), stride=(stride, 1)).squeeze(-1)
    print(f"x_unf shape1: {x_unf.shape}")  # [B, C_in*K, W_out]
    x_unf = x_unf.transpose(1, 2)  # [B, W_out, C_in*K]
    print(f"x_unf shape2: {x_unf.shape}")  # [B, W_out, C_in*K]

    if use_act_quant:
        scale_x = x_unf.abs().max() / 127 * delta
        # x_unf_int8 = ik.sym_quant_int8(x_unf, scale_x)
        x_unf_int8 = torch.clamp((x_unf / scale_x).round(), -128, 127).to(torch.int8)
    else:
        scale_x = 1.0
        x_unf_int8 = x_unf
    print(f"x_unf_int8 shape: {x_unf_int8.shape}")  # [B, W_out, C_in*K]

    weight_flat = weight_bf16.view(weight_bf16.shape[0], -1)  # [C_out, C_in*K]
    if use_weight_quant:
        scale_w = weight_flat.abs().max() / 127 * delta
        # weight_int8 = ik.sym_quant_int8(weight_flat, scale_w)
        weight_int8 = torch.clamp((weight_flat / scale_w).round(), -128, 127).to(torch.int8)
        import ipdb; ipdb.set_trace()
    else:
        scale_w = 1.0
        weight_int8 = weight_flat
    import ipdb; ipdb.set_trace()

    if use_act_quant and use_weight_quant:
        print("use int8 matmul")
        out_int32 = ik.matmul_int8(x_unf_int8.contiguous(), weight_int8.contiguous())
        out_int32 = out_int32.permute(0, 2, 1)
    else:
        # out_int32 = x_unf_int8.contiguous() @ weight_int8.T.contiguous().permute(0, 2, 1)
        print("use normal matmul")
        import ipdb; ipdb.set_trace()
        normal_conv = nn.Conv1d(x_bf16.shape[1], weight_bf16.shape[0], kernel_size, stride=stride, padding=padding, bias=False)
        normal_conv.weight.data = weight_bf16
        out_int32 = normal_conv(x_bf16)

    return out_int32, scale_x, scale_w

def linear_int8_fwd(x_bf16, weight_bf16, bias=None, use_act_quant=True, use_weight_quant=True):
    
    print(f"linear_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    in_features = x_bf16.shape[-1]
    out_features = weight_bf16.shape[0]
    # assert in_features == weight_bf16.shape[1], "Input and weight dimensions do not match

    if use_act_quant:
        scale_x = x_bf16.abs().max() / 127 * delta
        x_int8 = torch.clamp((x_bf16 / scale_x).round(), -128, 127).to(torch.int8)
    else:
        scale_x = 1.0
        x_int8 = x_bf16

    if use_weight_quant:
        scale_w = weight_bf16.abs().max() / 127 * delta
        weight_int8 = torch.clamp((weight_bf16 / scale_w).round(), -128, 127).to(torch.int8)
    else:
        scale_w = 1.0
        weight_int8 = weight_bf16

    if use_act_quant and use_weight_quant:
        out_int32 = ik.matmul_int8(x_int8.contiguous(), weight_int8.contiguous())
    else:
        # out_int32 = x_int8.contiguous() @ weight_int8.T.contiguous()
        print("use normal matmul")
        import ipdb; ipdb.set_trace()
        normallinear = nn.Linear(in_features, out_features, bias=False)
        normallinear.weight.data = weight_bf16
        out_int32 = normallinear(x_bf16)

    return out_int32, scale_x, scale_w