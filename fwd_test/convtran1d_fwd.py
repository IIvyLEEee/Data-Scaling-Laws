# import torch
# import torch.nn.functional as F
# import int4_kernel as ik  # 自定义int8 matmul kernel

# # ===================== 1. 原始输入 =====================
# x_bf16 = torch.randn(6, 768, 256).to(dtype=torch.bfloat16).cuda()  # [B, C_in, W_in]

# # ===================== 2. 卷积反转权重准备 =====================
# # convtranspose1d weight: [C_in, C_out, K]，注意顺序与 conv1d 不同
# weight_bf16 = torch.randn(768, 4, 3).to(torch.bfloat16).cuda()  # [C_in, C_out, K]
# weight_flip = weight_bf16.flip(-1)  # convtranspose 的核会 flip
# weight_flat = weight_flip.permute(1, 0, 2).reshape(4, -1)  # [C_out, C_in*K]

# # ===================== 3. 量化 =====================
# # scale_w = weight_flat.abs().max() / 127
# # weight_int8 = (weight_flat / scale_w).round().clamp(-128, 127).to(torch.int8)
# weight_int8 = weight_flat # [C_out, C_in*K]

# # 输出尺寸（公式）：W_out = (W_in - 1) * stride - 2*padding + kernel_size
# # 下面我们使用 stride=1, padding=1, kernel_size=3  => W_out = W_in
# W_in = x_bf16.shape[-1]
# kernel_size = 3
# padding = 1
# stride = 1
# W_out = (W_in - 1) * stride - 2 * padding + kernel_size

# # ===================== 4. im2col模拟（通过 F.pad + unfold） =====================
# # unfold 是用在 forward 的 conv1d；convtranspose1d 对应 fold
# x_reshape = x_bf16  # [B, C_in, W_in]
# x_unf = x_reshape  # [B, C_in, W_in]

# # 量化输入
# # scale_x = x_unf.abs().max() / 127
# # x_int8 = (x_unf / scale_x).round().clamp(-128, 127).to(torch.int8)  # [B, C_in, W_in]
# x_int8 = x_unf  # [B, C_in, W_in]

# # unfold-like操作：展开为滑动窗口输入（为 matmul 做准备）
# # x_unf_all = F.unfold(x_int8.unsqueeze(-1).float(), kernel_size=(1, kernel_size), dilation=(1, 1),
# #                      padding=(0, kernel_size - 1 - padding), stride=(1, stride)).to(torch.int8)
# x_unf_all = F.unfold(x_int8.unsqueeze(-1).float(), kernel_size=(1, kernel_size), dilation=(1, 1),
#                      padding=(0, kernel_size - 1 - padding), stride=(1, stride)).to(torch.bfloat16)
# # x_unf_all: [B, C_in*K, W_out]，与 convtranspose1d 对齐
# x_unf_all = x_unf_all.transpose(1, 2).contiguous()  # [B, W_out, C_in*K]

# # ===================== 5. 矩阵乘法 int8 =====================
# # out_int32 = ik.matmul_int8(x_unf_all, weight_int8.contiguous())  # [B, W_out, C_out]
# out_int32 = x_unf_all @ weight_int8.t()  # [B, W_out, C_out]

# # ===================== 6. 反量化 + 转置 =====================
# # scale_out = scale_x * scale_w
# # out_fp32 = out_int32.to(torch.float32) * scale_out
# out_fp32 = out_int32.to(torch.float32)  # [B, W_out, C_out]
# out_bf16 = out_fp32.permute(0, 2, 1).to(torch.bfloat16)  # [B, C_out, W_out]

# # ===================== 7. 验证 torch.conv_transpose1d =====================
# out_torch = F.conv_transpose1d(x_bf16, weight_bf16, stride=stride, padding=padding)
# print("Output from int8 convtranspose1d (bf16):", out_bf16.shape)
# print("Output from torch conv_transpose1d:", out_torch.shape)
# print("Output from int8 convtranspose1d (bf16):", out_bf16[0, 0, :])
# print("Output from torch conv_transpose1d:", out_torch[0, 0, :])

# # ===================== 8. 误差对比 =====================
# print("Max abs diff:", (out_bf16 - out_torch).abs().max().item())

import torch
import torch.nn.functional as F
import int4_kernel as ik

# 1. 原始输入 (模拟 ConvTranspose1d 的输入)
x_bf16_transpose = torch.randn(6, 8, 512).to(dtype=torch.bfloat16).cuda()  # [B, C_in, W_in]

# 2. 权重准备 (模拟 Conv1d 的 kernel，但用于转置卷积)
weight_bf16_transpose = torch.randn(8, 768, 3).to(torch.bfloat16).cuda()  # [C_out, C_in, K]
C_out, C_in, K = weight_bf16_transpose.shape

# 3. 模拟 ConvTranspose1d 的 unfold 的逆过程
# ConvTranspose1d with stride=1, padding=0 roughly corresponds to
# inserting zeros between input elements and then convolving.
# Here, we directly prepare the unfolded input that would produce the
# output of ConvTranspose1d.

W_in = x_bf16_transpose.shape[-1]
stride = 1
padding = 0
output_padding = 0
W_out = (W_in - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
print(f"Simulated W_out for ConvTranspose1d: {W_out}")

# 模拟在 W 维度上插入 stride - 1 个 0
x_upsampled = F.pad(x_bf16_transpose.unsqueeze(-1), (0, 0, 0, stride - 1), value=0).view(x_bf16_transpose.shape[0], x_bf16_transpose.shape[1], -1)

# 为了模拟 unfold 的逆过程，我们需要将 upsampled 的输入展开成与卷积核对应的形状
# 这里的逻辑需要仔细考虑 ConvTranspose1d 的操作
# 一种方法是直接构建目标输出形状，然后逆向填充

target_unfolded_shape = (x_bf16_transpose.shape[0], W_out, C_in * K)
x_unf_transpose = torch.zeros(target_unfolded_shape, dtype=torch.bfloat16).cuda()

for b in range(x_bf16_transpose.shape[0]):
    for c in range(C_in):
        for i in range(W_in):
            for k in range(K):
                out_w_index = i * stride + k
                if 0 <= out_w_index < W_out:
                    unf_index = i * K + k
                    x_unf_transpose[b, out_w_index, c * K + k] = x_bf16_transpose[b, c, i]


# 4. 量化
scale_x_transpose = x_unf_transpose.abs().max() / 127
x_unf_int8_transpose = (x_unf_transpose / scale_x_transpose).round().clamp(-128, 127).to(torch.int8).contiguous()

weight_flat_transpose = weight_bf16_transpose.view(C_out, -1)  # [C_out, C_in*K]
scale_w_transpose = weight_flat_transpose.abs().max() / 127
weight_int8_transpose = (weight_flat_transpose / scale_w_transpose).round().clamp(-128, 127).to(torch.int8)

# 执行矩阵乘法
print("x_unf_int8_transpose shape:", x_unf_int8_transpose.shape)
print("weight_int8_transpose shape:", weight_int8_transpose.shape)

out_int32_transpose = ik.matmul_int8(x_unf_int8_transpose, weight_int8_transpose.contiguous())

# 7. 反量化
scale_out_transpose = scale_x_transpose * scale_w_transpose
out_fp32_transpose = out_int32_transpose.to(torch.float32) * scale_out_transpose
out_bf16_transpose = out_fp32_transpose.permute(0, 2, 1).to(dtype=torch.bfloat16)  # [B, C_out, W_out]

print("Output bf16 using int8 (ConvTranspose1d):")
print(out_bf16_transpose.dtype)
print(out_bf16_transpose.shape)
print(out_bf16_transpose)

# compare with torch conv_transpose1d
out_torch_transpose = F.conv_transpose1d(x_bf16_transpose, weight_bf16_transpose, stride=1, padding=0)
print("Output bf16 using torch conv_transpose1d:")
print(out_torch_transpose.dtype)
print(out_torch_transpose.shape)
print(out_torch_transpose)

print("Difference (ConvTranspose1d):")
print(torch.abs(out_bf16_transpose - out_torch_transpose).max().item())