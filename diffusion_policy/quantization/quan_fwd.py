import torch
import torch.nn as nn
import torch.nn.functional as F
import int4_kernel as ik
from torch.nn.modules.utils import _pair
from logging.handlers import RotatingFileHandler
import logging

"""
Notice:
    x and weight are expected to be in bf16 format.
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

delta = 2.0

def conv1d_int8_fwd(x_bf16, weight_bf16, kernel_size=3, stride=1, padding=0, \
                    bias=None, dilation=1, groups=1, use_act_quant=True, use_weight_quant=True):
    assert groups == 1, "Only groups=1 is supported in this implementation"
    # print(f"conv1d_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    
    padding = padding[0]
    stride = stride[0]
    kernel_size = weight_bf16.shape[2]
    threshold = x_bf16.shape[1] * kernel_size
    if not use_act_quant or not use_weight_quant or (threshold<512):
        torch.cuda.synchronize()
        start = time.time()
        normal_conv = nn.Conv1d(x_bf16.shape[1], weight_bf16.shape[0], kernel_size, stride=stride, padding=padding, bias=False)
        normal_conv.weight.data = weight_bf16
        out_int32 = normal_conv(x_bf16)
        scale_x = 1.0
        scale_w = 1.0
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[conv1d] normal conv time: {end - start:.4f}s")

    else: # use_act_quant and use_weight_quant

        torch.cuda.synchronize()
        start_all = time.time()
        start = time.time()
        x_padded = F.pad(x_bf16.unsqueeze(-1), pad=(0, 0, padding, padding)) # [B, C_in, W_in+2*padding, 1]
        x_unf = F.unfold(x_padded, kernel_size=(kernel_size, 1), stride=(stride, 1)).squeeze(-1)
        x_unf = x_unf.transpose(1, 2)  # [B, W_out, C_in*K]
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[conv1d] unfold x time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        scale_x = x_unf.abs().max() / 127 * delta
        x_unf_int8 = torch.clamp((x_unf / scale_x).round(), -128, 127).to(torch.int8)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[conv1d] scale & quant x time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        weight_flat = weight_bf16.view(weight_bf16.shape[0], -1)  # [C_out, C_in*K]
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[conv1d] unfold weight time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        scale_w = weight_flat.abs().max() / 127 * delta
        weight_int8 = torch.clamp((weight_flat / scale_w).round(), -128, 127).to(torch.int8)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[conv1d] scale & quant weight time: {end - start:.4f}s")

        # logger.info(f"[conv1d] x_unf_int8.shape: {x_unf_int8.shape}, weight_int8.shape: {weight_int8.shape}")
        out_int32 = ik.matmul_int8(x_unf_int8.contiguous(), weight_int8.contiguous()).permute(0, 2, 1)
        torch.cuda.synchronize()
        end_all = time.time()
        logger.info(f"[conv1d] int8 conv time: {end_all - start_all:.4f}s")

    return out_int32, scale_x, scale_w

def linear_int8_fwd(x_bf16, weight_bf16, bias=None, use_act_quant=True, use_weight_quant=True):
    
    # print(f"linear_int8_fwd: x_bf16.shape: {x_bf16.shape}, weight_bf16.shape: {weight_bf16.shape}")
    in_features = x_bf16.shape[-1]
    out_features = weight_bf16.shape[0]
    # assert in_features == weight_bf16.shape[1], "Input and weight dimensions do not match

    threshold = x_bf16.shape[1]
    if not use_act_quant or not use_weight_quant or (threshold<512):
        torch.cuda.synchronize()
        start = time.time()
        normallinear = nn.Linear(in_features, out_features, bias=False)
        normallinear.weight.data = weight_bf16
        out_int32 = normallinear(x_bf16)
        scale_x = 1.0
        scale_w = 1.0
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[linear] normal linear time: {end - start:.4f}s")

    else: # use_act_quant and use_weight_quant
        torch.cuda.synchronize()
        start_all = time.time()
        start = time.time()
        scale_x = x_bf16.abs().max() / 127 * delta
        x_int8 = torch.clamp((x_bf16 / scale_x).round(), -128, 127).to(torch.int8)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[linear] scale & quant x time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        scale_w = weight_bf16.abs().max() / 127 * delta
        weight_int8 = torch.clamp((weight_bf16 / scale_w).round(), -128, 127).to(torch.int8)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"[linear] scale & quant weight time: {end - start:.4f}s")

        out_int32 = ik.matmul_int8(x_int8.contiguous(), weight_int8.contiguous())
        torch.cuda.synchronize()
        end_all = time.time()
        logger.info(f"[linear] int8 linear time: {end_all - start_all:.4f}s")

    return out_int32, scale_x, scale_w