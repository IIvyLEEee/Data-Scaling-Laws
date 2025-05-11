import torch
import torch.nn as nn
import torch.nn.functional as F
import int4_kernel as ik
from torch.nn.modules.utils import _pair
from logging.handlers import RotatingFileHandler
import logging

"""
Notice:
    x and weight are expected to be in int8/bf16 format.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 使用 RotatingFileHandler 防止日志文件过大
file_handler = RotatingFileHandler(
    'matrix_size.log', 
    maxBytes=5 * 1024 * 1024,  # 每个日志文件最大 5MB
    backupCount=3              # 保留 3 个备份日志
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

### Attention: this fwd do not consider bias!! bias is added in quant_layer foward!!

def conv1d_int8_fwd(x_int8, weight_flat, kernel_size=3, stride=1, padding=0, \
                    bias=None, dilation=1, groups=1):
    assert groups == 1, "Only groups=1 is supported in this implementation"
    # print(f"conv1d_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    # print(f"x_int8: {x_int8.dtype}")
    # print(f"x_int8: {x_int8.shape}")
    if x_int8.dtype is not torch.int8:
        # print(f"weight_flat: {weight_flat.dtype}")
        x_bf16 = x_int8.to(torch.bfloat16)
        weight_bf16 = weight_flat.to(torch.bfloat16)
        out_int32 = F.conv1d(x_bf16, weight_bf16, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    else:
        assert x_int8.dtype == torch.int8
        assert weight_flat.dtype == torch.int8

        padding = padding[0]
        stride = stride[0]
        kernel_size = weight_flat.shape[1] // x_int8.shape[1]
        # print(f"kernel size: {kernel_size}")

        x_bf16 = x_int8.to(torch.bfloat16)

        x_padded = F.pad(x_bf16.unsqueeze(-1), pad=(0, 0, padding, padding)) # [B, C_in, W_in+2*padding, 1]
        x_unf = F.unfold(x_padded, kernel_size=(kernel_size, 1), stride=(stride, 1)).squeeze(-1)
        x_unf = x_unf.transpose(1, 2)  # [B, W_out, C_in*K]
        x_unf_int8 = x_unf.to(torch.int8)  # [B, W_out, C_in*K]

        # weight_flat = weight_int8.view(weight_int8.shape[0], -1)  # [C_out, C_in*K]

        out_int32 = ik.matmul_int8(x_unf_int8.contiguous(), weight_flat.contiguous()).permute(0, 2, 1)

    return out_int32

def linear_int8_fwd(x_int8, weight_int8, bias=None):
    
    # print(f"linear_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    # print(f"x_int8: {x_int8.dtype}")
    # print(f"x_int8: {x_int8.shape}")
    if x_int8.dtype is not torch.int8:
        # print(f"weight_int8: {weight_int8.dtype}")
        x_bf16 = x_int8.to(torch.bfloat16)
        weight_bf16 = weight_int8.to(torch.bfloat16)
        out_int32 = F.linear(x_bf16, weight_bf16, bias=None)
    
    else:
        assert x_int8.dtype == torch.int8
        assert weight_int8.dtype == torch.int8

        out_int32 = ik.matmul_int8(x_int8.contiguous(), weight_int8.contiguous())
        
    return out_int32