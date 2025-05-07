import torch
import torch.nn.functional as F
import int4_kernel as ik
# from diffusion_policy.quantization.quan_layer import Activation_Quantizer
# from diffusion_policy.quantization.quan_layer import Weight_Quantizer

"""
Notice:
    x and weight are expected to be in bf16 format.
"""

# aq = Activation_Quantizer()
# wq = Weight_Quantizer()
delta = 2.0

def conv1d_int8_fwd(x_bf16, weight_bf16, kernel_size, stride=1, padding=0, \
                    bias=None, dilation=1, groups=1, use_act_quant=True, use_weight_quant=True):

    assert groups == 1, "Only groups=1 is supported in this implementation"

    x_padded = F.pad(x_bf16.unsqueeze(-1), pad=(0, 0, padding, padding))
    x_unf = F.unfold(x_padded, kernel_size=(kernel_size, 1), stride=(1, 1)).squeeze(-1)
    # x_unf: [B, C_in*K, W_out]  =>  transpose to [B, W_out, C_in*K]
    x_unf = x_unf.transpose(1, 2)  # [B, W_out, C_in*K]

    if use_act_quant:
        scale_x = x_unf.abs().max() / 127 * delta
        x_unf_int8 = ik.sym_quant_int8(x_unf, scale_x)
    else:
        scale_x = 1.0
        x_unf_int8 = x_unf

    weight_flat = weight_bf16.view(weight_bf16.shape[0], -1)  # [C_out, C_in*K]
    # scale_w = weight_flat.abs().max() / 127 * delta
    # weight_int8 = ik.sym_quant_int8(weight_flat, scale_w)
    if use_weight_quant:
        scale_w = weight_flat.abs().max() / 127 * delta
        weight_int8 = ik.sym_quant_int8(weight_flat, scale_w)
    else:
        scale_w = 1.0
        weight_int8 = weight_flat

    if use_act_quant and use_weight_quant:
        out_int32 = ik.matmul_int8(x_unf_int8.contiguous(), weight_int8.contiguous()).permute(0, 2, 1)
    else:
        out_int32 = x_unf_int8.contiguous() @ weight_int8.T.contiguous().permute(0, 2, 1)

    return out_int32, scale_x, scale_w

def linear_int8_fwd(x_bf16, weight_bf16, bias=None, use_act_quant=True, use_weight_quant=True):
    
    in_features = x_bf16.shape[-1]
    out_features = weight_bf16.shape[0]

    if use_act_quant:
        scale_x = x_bf16.abs().max() / 127 * delta
        x_int8 = ik.sym_quant_int8(x_bf16, scale_x)
    else:
        scale_x = 1.0
        x_int8 = x_bf16

    if use_weight_quant:
        scale_w = weight_bf16.abs().max() / 127 * delta
        weight_int8 = ik.sym_quant_int8(weight_bf16, scale_w)
    else:
        scale_w = 1.0
        weight_int8 = weight_bf16

    if use_act_quant and use_weight_quant:
        out_int32 = ik.matmul_int8(x_int8.contiguous(), weight_int8.contiguous())
    else:
        out_int32 = x_int8.contiguous() @ weight_int8.T.contiguous()

    return out_int32, scale_x, scale_w