import torch
import torch.nn.functional as F
import int4_kernel as ik

"""
Notice:
    x and weight are expected to be in int8 format.
"""


def conv1d_int8_fwd(x_unf, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    使用 unfold + 自定义 matmul_int8 实现 int8 conv1d 的前向传播
    x: [N, C_in, L]
    weight: [C_out, C_in/groups, K]
    x_unf: [N, L_out, C_in * K]
    A: [N * L_out, C_in * K]
    B: [C_out, C_in * K]
    output: [N * L_out, C_out]
    """

    assert groups == 1, "Only groups=1 is supported in this implementation"

    N, C_ink, L_in = x_unf.shape
    C_out, _, K = weight.shape
    C_in = C_ink // K

    # Step 1: unfold input to shape [N, C_in*K, L_out]
    print(x_unf.device)
    # x_unf shape: [N, C_in*K, L_out]
    x_unf = x_unf.transpose(1, 2)  # [N, L_out, C_in*K]
    x_unf = x_unf.reshape(-1, C_in * K)  # [N*L_out, C_in*K]

    # Step 2: reshape weight to [C_out, C_in*K]
    weight_flat = weight.view(C_out, -1)  # [C_out, C_in*K]

    # Step 3: call your custom matmul_int8(A, B)
    # 注意，这里的 A 是输入 [N*L_out, C_in*K]，B 是 weight [C_out, C_in*K]
    # 所以需要先转置 B 为 [C_in*K, C_out]，再转置回输出
    A = x_unf.contiguous()
    B = weight_flat.contiguous()
    print(A.shape)
    print(B.shape)
    output = ik.matmul_int8(A, B.contiguous())  # output: [N*L_out, C_out]

    print(output.shape)
    # Step 4: reshape输出为 [N, C_out, L_out]
    L_out = output.shape[0] // N
    output = output.view(N, L_out, C_out).permute(0, 2, 1).contiguous()

    # Step 5: 加 bias（可选）
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output

def convtranspose1d_int8_fwd(x_unf, Lin, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    """
    x: [N, C_in, L_in]
    weight: [C_in, C_out // groups, K]
    返回: [N, C_out, L_out]
    """
    N, C_ink, L_in = x_unf.shape
    L_in = Lin
    C_out = weight.shape[1] * groups
    K = weight.shape[2]
    C_in = C_ink // K

    # 计算输出长度（PyTorch的公式）
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    print(f"L_out = ({L_in} - 1) * {stride} - 2 * {padding} + {dilation} * ({K} - 1) + {output_padding} + 1 = {L_out}")
    # 3 + 1 * 3 + 1 = 7
    # # Step 1: 上采样 input（插入 0）
    # x_upsampled = torch.zeros((N, C_in, (L_in - 1) * stride + 1), dtype=torch.int8, device=x.device)
    # x_upsampled[:, :, ::stride] = x  # 插入零点

    # # Step 2: padding
    # x_padded = F.pad(x_upsampled, (padding, padding))

    # # Step 3: unfold 成 [N, C_in*K, L_out]
    # x_unf = F.unfold(
    #     x_padded.unsqueeze(-1),  # [N, C_in, L, 1]
    #     kernel_size=(K, 1),
    #     dilation=(dilation, 1),
    #     padding=(0, 0),
    #     stride=(1, 1)
    # )  # → [N, C_in*K, L_out]
    print(x_unf.shape)
    x_unf = x_unf.transpose(1, 2).reshape(-1, C_in * K)  # [N*L_out, C_in*K]

    # Step 4: reshape权重为 [C_out, C_in*K]
    weight_flat = weight.permute(1, 0, 2).reshape(C_out, -1)  # [C_out, C_in*K]

    print("in convtranspose1d_int8_fwd")
    print(x_unf.shape)
    print(weight_flat.shape)

    # Step 5: matmul
    output = ik.matmul_int8(x_unf, weight_flat.contiguous())  # [N*L_out, C_out]
    print(output.shape)
    output = output.view(N, L_out, C_out).permute(0, 2, 1).contiguous()  # [N, C_out, L_out]

    # Step 6: bias（可选）
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output

def linear_int8_fwd(x, weight, bias=None):
    """
    使用自定义 matmul_int8 实现 int8 linear 的前向传播
    x: [N, C_in]
    weight: [C_out, C_in]
    """
    assert x.dtype == torch.int8 and weight.dtype == torch.int8, "x and weight must be int8"
    assert x.device == weight.device, "x and weight must be on the same device"
    if bias is not None:
        assert bias.device == x.device, "bias must be on the same device as x"

    A = x.contiguous()
    B = weight.contiguous()
    output = ik.matmul_int8(A, B)  # output: [N, C_out]

    # Step 2: 加 bias（可选）
    if bias is not None:
        output += bias.view(1, -1)

    return output



if __name__ == "__main__":
    from diffusion_policy.quantization.quan_layer import Activation_Quantizer

    aq = Activation_Quantizer()

    # 设置随机种子以便结果可复现
    torch.manual_seed(0)

    # 测试参数
    N, C_in, L_in = 16, 32, 4  # Batch size, 输入通道数, 输入长度
    C_out = 32  # 输出通道数
    K = 3  # 卷积核大小
    stride = 1
    padding = 0
    dilation = 1
    output_padding = 0

    # 生成随机输入数据、权重和偏置
    x = torch.randint(-128, 128, (N, C_in, L_in), dtype=torch.int8).cuda()
    x_bf16 = x.to(torch.bfloat16)
    x_unf_bf16 = F.unfold(x_bf16.unsqueeze(-1), kernel_size=(K, 1),
                     dilation=(dilation, 1),
                     padding=(padding, 0),
                     stride=(stride, 1))
    scale, zero_point = aq.init_quantization_scale(x_unf_bf16)
    print(x_unf_bf16.shape)
    print(scale.shape)
    scale_expanded = torch.full((x_unf_bf16.shape[0], x_unf_bf16.shape[1], 1), scale).to(torch.bfloat16).cuda()
    print(scale_expanded.shape)
    print(x_unf_bf16.dtype)
    print(scale_expanded.dtype)
    x_unf = ik.sym_quant_int8(x_unf_bf16, scale_expanded)
    L_in = x_unf.shape[2]

    if stride == 1:
        x_upsampled = x_unf.clone()
    else:
        L_upsampled = (L_in - 1) * stride + 1
        x_upsampled = torch.zeros((N, C_in, L_upsampled), dtype=x.dtype, device=x.device)
        x_upsampled[:, :, ::stride] = x_unf

    # # Step 2: padding
    x_padded = F.pad(x_upsampled, (padding, padding)).to(torch.bfloat16)
    print(f"x_padded shape: {x_padded.shape}")

    # # Step 3: unfold 成 [N, C_in*K, L_out]
    # xp_unf_bf16 = F.unfold(
    #     x_padded.unsqueeze(-1),  # [N, C_in, L, 1]
    #     kernel_size=(K, 1),
    #     dilation=(dilation, 1),
    #     padding=(0, 0),
    #     stride=(1, 1)
    # )  # → [N, C_in*K, L_out]

    # scale, zero_point = aq.init_quantization_scale(xp_unf_bf16)
    # scale_expanded = torch.full((xp_unf_bf16.shape[0], xp_unf_bf16.shape[1], 1), scale).to(torch.bfloat16).cuda()
    # xp_unf = ik.sym_quant_int8(xp_unf_bf16, scale_expanded)

    weight_conv1d = torch.randint(-128, 128, (C_out, C_in, K), dtype=torch.int8).cuda()
    weight_convtranspose1d = torch.randint(-128, 128, (C_in, C_out, K), dtype=torch.int8).cuda()
    weight_linear = torch.randint(-128, 128, (C_out, C_in), dtype=torch.int8).cuda()

    bias_conv1d = torch.randint(-128, 128, (C_out,), dtype=torch.int8).cuda()
    bias_convtranspose1d = torch.randint(-128, 128, (C_out,), dtype=torch.int8).cuda()
    bias_linear = torch.randint(-128, 128, (C_out,), dtype=torch.int8).cuda()

    # 使用自定义函数进行计算
    conv1d_output = conv1d_int8_fwd(x_unf, weight_conv1d, bias=bias_conv1d, stride=stride, padding=padding, dilation=dilation)
    # convtranspose1d_output = convtranspose1d_int8_fwd(xp_unf, L_in, weight_convtranspose1d, bias=bias_convtranspose1d, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
    # linear_output = linear_int8_fwd(x.view(N, -1), weight_linear, bias=bias_linear)

    # 使用 PyTorch 的实现进行计算
    conv1d_pytorch = F.conv1d(x.float(), weight_conv1d.float(), bias=bias_conv1d.float() if bias_conv1d is not None else None, stride=stride, padding=padding, dilation=dilation)
    # convtranspose1d_pytorch = F.conv_transpose1d(x.float(), weight_convtranspose1d.float(), bias=bias_convtranspose1d.float() if bias_convtranspose1d is not None else None, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
    # linear_pytorch = F.linear(x.float(), weight_linear.float(), bias=bias_linear.float() if bias_linear is not None else None)

    # 比较结果差异
    print("Conv1d Int8 Forward vs PyTorch Conv1d:")
    print("Difference:", torch.abs(conv1d_output - conv1d_pytorch).sum().item())
    # print("ConvTranspose1d Int8 Forward vs PyTorch ConvTranspose1d:")
    print("Difference:", torch.abs(convtranspose1d_output - convtranspose1d_pytorch).sum().item())
    print("Linear Int8 Forward vs PyTorch Linear:")
    print("Difference:", torch.abs(linear_output - linear_pytorch).sum().item())
