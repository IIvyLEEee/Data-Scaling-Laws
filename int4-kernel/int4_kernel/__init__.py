import torch
import int4_kernel._CUDA

__all__ = [ 
  "matmul", #int-4 matmul
  "sym_quant", "sym_dequant", "PackedQuantizedTensor", # Quantization
  "sym_dequant_row_only", "sym_dequant_col_only", "pack_i4" # Quantization
  "sym_dequantize_quantize", "matmul_int8" # Quantization
]

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)

def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last

def flatten_first_dim_and_return_shape(x: torch.Tensor):
    shape_excl_first = x.shape[1:]
    x = x.view(x.shape[0], -1)
    return x, shape_excl_first

def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return int4_kernel._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def matmul_int8(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return int4_kernel._CUDA.matmul_int8(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.bfloat16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return int4_kernel._CUDA.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_quant_int8(x, scale):
    assert x.dtype == scale.dtype == torch.bfloat16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return int4_kernel._CUDA.sym_quant_int8(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_quant_int2(x, scale):
    assert x.dtype == scale.dtype == torch.bfloat16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return int4_kernel._CUDA.sym_quant_int2(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_quant_int2_col(x, scale):
    assert x.dtype == scale.dtype == torch.bfloat16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return int4_kernel._CUDA.sym_quant_int2_col(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)

def sym_dequant_row_only(q, scale_row, bits=32):
    assert q.dtype == torch.int8
    assert scale_row.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_row_only(q, scale_row.view(-1), bits).view(*q_shape_excl_last, -1)

def sym_dequant_row_only_int8(q, scale_row):
    assert q.dtype == torch.int8
    assert scale_row.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_row_only_int8(q, scale_row.view(-1)).view(*q_shape_excl_last, -1)

def sym_dequant_row_only_int2(q, scale_row):
    assert q.dtype == torch.int8
    assert scale_row.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_row_only_int2(q, scale_row.view(-1)).view(*q_shape_excl_last, -1)

def sym_dequant_col_only_int2(q, scale_col):
    assert q.dtype == torch.int8
    assert scale_col.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_first_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_col_only_int2(q, scale_col.view(-1)) #.view(*q_shape_excl_last, -1)

def sym_dequant_col_only(q, scale_col, bits=32):
    assert q.dtype == torch.int8
    assert scale_col.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_first_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_col_only(q, scale_col.view(-1), bits).view(-1, *q_shape_excl_last)

def sym_dequantize_quantize(q_in, scale_row, scale_col):
    assert q_in.dtype == torch.int8
    assert scale_row.dtype == scale_col.dtype == torch.bfloat16
    q_in, q_in_shape_excl_last = flatten_last_dim_and_return_shape(q_in)
    return int4_kernel._CUDA.sym_dequantize_quantize(q_in, scale_row, scale_col) #.view(*q_in_shape_excl_last, -1)

def int4_to_int8(q):
    assert q.dtype == torch.int8
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.int4_to_int8(q).view(*q_shape_excl_last, -1)

def pack_i4(q, bits=4):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    maxq = torch.tensor(2**(bits-1)-1)
    minq = torch.tensor(-maxq - 1)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq)), f'The tensor max value is {torch.max(q)} and min value is {torch.min(q)}'

    # q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i8 = q.to(dtype=torch.int8)
    q_i8 = torch.where(q_i8 < 0, 2 ** bits + q_i8, q_i8).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4

class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
