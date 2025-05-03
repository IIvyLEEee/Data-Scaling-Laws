import torch
import int4_kernel

if __name__ == '__main__':
  # randomly set seed
  torch.manual_seed(0)
  x = torch.randn((2, 512, 1024), dtype=torch.bfloat16, device='cuda')
  x_shape = x.shape
  x = x.view(-1, x_shape[-1]).contiguous()
  s = (torch.max(x, dim=-2)[0].unsqueeze(1) - torch.min(x, dim=-2)[0].unsqueeze(1)) / (2 ** 2 - 1) + torch.finfo(torch.bfloat16).smallest_normal 
  # quantize
  x_quant = int4_kernel.sym_quant_int2_col(x, s)
  # dequantize
  x_dequant = int4_kernel.sym_dequant_col_only_int2(x_quant, s)
  # print(x_dequant.abs().mean())
  # print(x.abs().mean())
  print(x)
  print(x_dequant)