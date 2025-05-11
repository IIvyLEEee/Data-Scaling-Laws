import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
import int4_kernel as ik
import random

random.seed(40)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# 1. 原始输入
# x_bf16 = tensor([[[1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.]]])  # [B, C_in, W_in]
x_bf16 = torch.randn(3, 768, 6).to(torch.float64).cuda()  # [B, C_in, W_in]
padding = 0
stride = 2
kernel_size = 3

# >>> 添加padding，模拟Conv1d中padding=1的效果 <<<
# Conv1d: padding=1 等价于在W维度左边加1，右边加1
# unsqueeze后形状为 [B, C_in, W_in, 1]
x_padded = F.pad(x_bf16.unsqueeze(-1), pad=(0, 0, padding, padding))  # pad=(left, right, top, bottom)
print("x_padded shape:", x_padded.shape)  # [B, C_in, W_in+2*padding, 1]

# 2. unfold 操作（仍然在 bf16 下完成）
# 模拟 conv1d: kernel_size=3, stride=1
x_unf = F.unfold(x_padded, kernel_size=(kernel_size, 1), stride=(stride, 1)).squeeze(-1)
print("x_unf shape:", x_unf.shape)  # [B, C_in*K, W_out]
# x_unf: [B, C_in*K, W_out]  =>  transpose to [B, W_out, C_in*K]
x_unf = x_unf.transpose(1, 2)  # [B, W_out, C_in*K]
print("x_unf shape after transpose:", x_unf.shape)  # [B, W_out, C_in*K]

# 3. 量化（使用简单的对称量化为例）
# scale_x = x_unf.abs().max() / 127
# x_unf_int8 = (x_unf / scale_x).round().clamp(-128, 127).to(torch.int8).contiguous()
x_unf_int8 = x_unf

# 4. 权重准备（以 conv1d kernel 为例）
weight_bf16 = torch.randn(1, 768, 3).to(torch.float64).cuda()  # [C_out, C_in, K]
# weight_bf16 = tensor([[2., 2., 2.], [2., 2., 2.]]).view(1, 2, 3)  # [C_out, C_in, K]
print("weight_bf16 shape:", weight_bf16.shape)  # [C_out, C_in, K]
output_channels = weight_bf16.shape[0]
input_channels = x_bf16.shape[1]
# kernel_size = weight_bf16.shape[1]
weight_torch = weight_bf16.view(output_channels, input_channels, kernel_size)  # [1, 1, 3]

weight_flat = weight_bf16.view(output_channels, -1)  # [C_out, C_in*K]
# scale_w = weight_flat.abs().max() / 127
# weight_int8 = (weight_flat / scale_w).round().clamp(-128, 127).to(torch.int8)
weight_int8 = weight_flat

# 执行
print("x_unf_int8 shape:", x_unf_int8.shape)
print("weight_int8 shape:", weight_int8.shape)

# out_int32 = ik.matmul_int8(x_unf_int8, weight_int8.contiguous())
out_int32 = x_unf_int8 @ weight_int8.T

# 7. 反量化（若需要回到float/bf16）
# scale_out = scale_x * scale_w
# out_fp32 = out_int32.to(torch.float32) * scale_out
# out_bf16 = out_fp32.permute(0, 2, 1).to(dtype=torch.bfloat16)  # [B, C_out, W_out]
out_bf16 = out_int32.permute(0, 2, 1)  # [B, C_out, W_out]

print("Output bf16 using int8:")
print(out_bf16.dtype)
print(out_bf16.shape)
print(out_bf16)

# compare with torch conv1d
# input_channels = x_bf16.shape[1]
# output_channels = weight_torch.shape[0]
print("output channels:", output_channels)
nnconv = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding,\
                     bias=False)
print("nnconv weight shape:", nnconv.weight.shape)
nnconv.weight.data = weight_torch

out_torch = nnconv(x_bf16)
print("Output bf16 using torch conv1d:")
print(out_torch.dtype)
print(out_torch.shape)
print(out_torch)

print("Difference:")
print(torch.abs(out_bf16 - out_torch).max().item())
