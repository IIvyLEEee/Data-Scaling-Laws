import torch
import int4_kernel

if __name__ == '__main__':
  # -----------------------------------------------------------------------
  # torch.set_printoptions(edgeitems=8, linewidth=100)
  # x = torch.randn((2, 512, 1024), dtype=torch.bfloat16, device='cuda')
  # w = torch.randn((2048, 1024), dtype=torch.bfloat16, device='cuda')
  
  # # fake quantize
  # x_scale = (torch.max(x, dim=-1)[0].unsqueeze(1) - torch.min(x, dim=-1)[0].unsqueeze(1)) / (2 ** 8 - 1) + torch.finfo(torch.bfloat16).smallest_normal 
  # x_scale = x_scale.transpose(-2, -1)
  # x_quantized_dequantized = torch.clamp(torch.round(x / x_scale), -2**7, 2**7-1) * x_scale
  
  # w_scale = (torch.max(w, dim=-1)[0].unsqueeze(1) - torch.min(w, dim=-1)[0].unsqueeze(1)) / (2 ** 8 - 1) + torch.finfo(torch.bfloat16).smallest_normal
  # w_quantized_dequantized = torch.clamp(torch.round(w / w_scale), -2**7, 2**7-1) * w_scale
  
  # y = (x_quantized_dequantized @ w_quantized_dequantized.T)
  
  # # quantize the x in main part
  # x_quantized = int4_kernel.sym_quant_int8(x, x_scale)
  # w_quantized = int4_kernel.sym_quant_int8(w, w_scale)
  
  # y_quantized = int4_kernel.matmul_int8(x_quantized, w_quantized)
  # y_dequantized = int4_kernel.sym_dequant(y_quantized, x_scale, w_scale)
  
  # print('y:', y)
  # print('diff:', y - y_dequantized)
  
  # -----------------------------------------------------------------------
  torch.set_printoptions(edgeitems=8, linewidth=100)
  x = torch.randn((2, 512, 1024), dtype=torch.bfloat16, device='cuda')
  w = torch.randn((2048, 1024), dtype=torch.bfloat16, device='cuda')
  
  # fake quantize
  x_scale = (torch.max(x, dim=-1)[0].unsqueeze(1) - torch.min(x, dim=-1)[0].unsqueeze(1)) / (2 ** 8 - 1) + torch.finfo(torch.bfloat16).smallest_normal 
  x_scale = x_scale.transpose(-2, -1)
  x_quantized_dequantized = torch.clamp(torch.round(x / x_scale), -2**7, 2**7-1) * x_scale
  
  w_scale = (torch.max(w, dim=-1)[0].unsqueeze(1) - torch.min(w, dim=-1)[0].unsqueeze(1)) / (2 ** 4 - 1) + torch.finfo(torch.bfloat16).smallest_normal
  w_quantized_dequantized = torch.clamp(torch.round(w / w_scale), -2**3, 2**3-1) * w_scale
  
  y = (x_quantized_dequantized @ w_quantized_dequantized.T)
  
  # quantize the x in main part
  x_quantized = int4_kernel.sym_quant_int8(x, x_scale)
  w_quantized = int4_kernel.sym_quant(w, w_scale)
  
  # convert int4 to int8
  w_quantized = int4_kernel.int4_to_int8(w_quantized)
  
  y_quantized = int4_kernel.matmul_int8(x_quantized, w_quantized)
  y_dequantized = int4_kernel.sym_dequant(y_quantized, x_scale, w_scale)
  
  print('y:', y)
  print('diff:', y - y_dequantized)
  
  # -----------------------------------------------------------------------
  x_dequantized = int4_kernel.sym_dequant_row_only_int8(x_quantized, x_scale)
  print('x - x_dequantized:', x - x_dequantized)