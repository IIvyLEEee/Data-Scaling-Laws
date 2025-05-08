import torch
import torch.nn.functional as F
import int4_kernel as ik

# 1. 原始输入 (模拟线性层的输入)
x_bf16_linear = torch.randn(6, 512).to(dtype=torch.float64).cuda()  # [B, in_features]

# 2. 权重准备 (线性层的权重)
in_features = x_bf16_linear.shape[-1]
out_features = 768
weight_bf16_linear = torch.randn(out_features, in_features).to(torch.float64).cuda()  # [out_features, in_features]

# 3. 量化输入
# scale_x_linear = x_bf16_linear.abs().max() / 127
# x_int8_linear = (x_bf16_linear / scale_x_linear).round().clamp(-128, 127).to(torch.int8).contiguous()
x_int8_linear = x_bf16_linear

# 4. 量化权重
# scale_w_linear = weight_bf16_linear.abs().max() / 127
# weight_int8_linear = (weight_bf16_linear / scale_w_linear).round().clamp(-128, 127).to(torch.int8).contiguous()
weight_int8_linear = weight_bf16_linear

# 执行矩阵乘法
print("x_int8_linear shape:", x_int8_linear.shape)
print("weight_int8_linear shape:", weight_int8_linear.shape)

# out_int32_linear = ik.matmul_int8(x_int8_linear, weight_int8_linear.contiguous())  # 注意权重要转置
out_int32_linear = x_int8_linear @ weight_int8_linear.T.contiguous()  # [B, out_features]
print("out_int32_linear shape:", out_int32_linear.shape)

# 5. 反量化
# scale_out_linear = scale_x_linear * scale_w_linear
# out_fp32_linear = out_int32_linear.to(torch.float32) * scale_out_linear
out_bf16_linear = out_int32_linear

print("Output bf16 using int8 (Linear):")
print(out_bf16_linear.dtype)
print(out_bf16_linear.shape)
print(out_bf16_linear)

# compare with torch linear
linear_layer = torch.nn.Linear(in_features, out_features).to(torch.float64).cuda()
with torch.no_grad():
    linear_layer.weight.copy_(weight_bf16_linear)  # 注意 PyTorch Linear 层的权重是 [out_features, in_features]
    linear_layer.bias.fill_(0)  # 假设没有偏置
out_torch_linear = linear_layer(x_bf16_linear)
print("Output bf16 using torch Linear:")
print(out_torch_linear.dtype)
print(out_torch_linear.shape)
print(out_torch_linear)

print("Difference (Linear):")
print(torch.abs(out_bf16_linear - out_torch_linear).max().item())