#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <quant.h>

torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * kElementsPerVector; // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K,
              C.data_ptr<int32_t>());

  return C;
}

torch::Tensor matmul_int8(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("matmul_int8", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul_int8", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul_int8", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_host_int8(A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
  torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0,
                   x.size(0));
  uint32_t rows = x.size(0);
  uint32_t colsSrc = x.size(1);
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

  auto q = torch::empty({rows, colsDst},
                        torch::dtype(torch::kInt8).device(x.device()));

  sym_quant_host((half_bf16 *)x.data_ptr(), (half_bf16 *)scale.data_ptr(), rows, colsSrc,
                 colsDst, q.data_ptr<Int4Storage>());

  return q;
}

torch::Tensor sym_quant_int8(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant_int8", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant_int8", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant_int8", {x, "x", 0}, {scale, "scale", 1});
  torch::checkSize("sym_quant_int8", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
  uint32_t rows = x.size(0);
  uint32_t cols = x.size(1);

  auto q = torch::empty({rows, cols},
                        torch::dtype(torch::kInt8).device(x.device()));

  sym_quant_int8_host((half_bf16 *)x.data_ptr(), (half_bf16 *)scale.data_ptr(), rows, cols, q.data_ptr<int8_t>());

  return q;
}

torch::Tensor sym_quant_int2(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant_int2", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant_int2", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant_int2", {x, "x", 0}, {scale, "scale", 1});

  uint32_t kElementsPerVector = 4;
  uint32_t rows = x.size(0);
  uint32_t colsSrc = x.size(1);
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

  auto q = torch::empty({rows, colsDst},
                        torch::dtype(torch::kInt8).device(x.device()));
  
  sym_quant_int2_host((half_bf16 *)x.data_ptr(), (half_bf16 *)scale.data_ptr(), rows, colsSrc, colsDst, (int8_t*)q.data_ptr());

  return q;
}

torch::Tensor sym_quant_int2_col(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant_int2_col", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant_int2_col", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant_int2_col", {x, "x", 0}, {scale, "scale", 1});

  uint32_t kElementsPerVector = 4;
  uint32_t rows = x.size(0);
  uint32_t colsSrc = x.size(1);
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

  auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kInt8).device(x.device()));
  
  sym_quant_int2_col_host((half_bf16 *)x.data_ptr(), (half_bf16 *)scale.data_ptr(), rows, colsSrc, colsDst, (int8_t*)q.data_ptr());

  return q;
}

torch::Tensor sym_dequant(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const torch::Tensor &scale_col, const int bits) {
  torch::checkAllContiguous(
      "sym_dequant",
      {{q, "q", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});
  torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU(
      "sym_dequant",
      {{q, "q", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});

  uint32_t rows = q.size(0);
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1},
                   0, rows);
  torch::checkSize("sym_dequant", torch::TensorArg{scale_col, "scale_col", 2},
                   0, cols);

  auto x =
      torch::empty(q.sizes(), torch::dtype(torch::kBFloat16).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_host(q.data_ptr<int32_t>(), (half_bf16 *)scale_row.data_ptr(),
                     (half_bf16 *)scale_col.data_ptr(), rows, cols,
                     (half_bf16 *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_col_only(const torch::Tensor &q,
                                   const torch::Tensor &scale_col, const int bits) {
  torch::checkAllContiguous("sym_dequant_col_only", {{q, "q", 0}, {scale_col, "scale_col", 1}});
  torch::checkDeviceType("sym_dequant_col_only", {q, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_col_only", {{q, "q", 0}, {scale_col, "scale_col", 1}});

  uint32_t rowsSrc = q.size(0);
  uint32_t rowsDst = rowsSrc * kElementsPerVector;
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant_col_only", torch::TensorArg{scale_col, "scale_col", 1},
                   0, cols);

  auto x =
      torch::empty({rowsDst, cols}, torch::dtype(torch::kBFloat16).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_col_only_host(q.data_ptr<int8_t>(), (half_bf16 *)scale_col.data_ptr(),
                              rowsSrc, rowsDst, cols, (half_bf16 *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_row_only(const torch::Tensor &q,
                                   const torch::Tensor &scale_row, const int bits) {
  torch::checkAllContiguous("sym_dequant_row_only", {{q, "q", 0}, {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_row_only", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_row_only", {{q, "q", 0}, {scale_row, "scale_row", 1}});

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  torch::checkSize("sym_dequant_row_only", torch::TensorArg{scale_row, "scale_row", 1},
                   0, rows);

  auto x =
      torch::empty({rows, colsDst}, torch::dtype(torch::kBFloat16).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_row_only_host(q.data_ptr<int8_t>(), (half_bf16 *)scale_row.data_ptr(),
                              rows, colsSrc, colsDst, (half_bf16 *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_row_only_int8(const torch::Tensor &q,
                                        const torch::Tensor &scale_row) {
  torch::checkAllContiguous("sym_dequant_row_only_int8", {{q, "q", 0}, {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_row_only_int8", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_row_only_int8", {{q, "q", 0}, {scale_row, "scale_row", 1}});

  uint32_t rows = q.size(0);
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant_row_only_int8", torch::TensorArg{scale_row, "scale_row", 1}, 0, rows);

  auto x =
      torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(q.device()));

  sym_dequant_row_only_int8_host(q.data_ptr<int8_t>(), (half_bf16 *)scale_row.data_ptr(),
                                 rows, cols, (half_bf16 *)x.data_ptr());

  return x;
}

torch::Tensor sym_dequant_row_only_int2(const torch::Tensor &q,
                                        const torch::Tensor &scale_row) {
  torch::checkAllContiguous("sym_dequant_row_only_int2", {{q, "q", 0}, {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_row_only_int2", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_row_only_int2", {{q, "q", 0}, {scale_row, "scale_row", 1}});

  int kElementsPerVector = 4;

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  torch::checkSize("sym_dequant_row_only_int2", torch::TensorArg{scale_row, "scale_row", 1}, 0, rows);

  auto x =
      torch::empty({rows, colsDst}, torch::dtype(torch::kBFloat16).device(q.device()));

  sym_dequant_row_only_int2_host(q.data_ptr<int8_t>(), (half_bf16 *)scale_row.data_ptr(),
                                 rows, colsSrc, colsDst, (half_bf16 *)x.data_ptr());

  return x;
}

torch::Tensor sym_dequant_col_only_int2(const torch::Tensor &q,
                                        const torch::Tensor &scale_col) {
  torch::checkAllContiguous("sym_dequant_col_only_int2", {{q, "q", 0}, {scale_col, "scale_col", 1}});
  torch::checkDeviceType("sym_dequant_col_only_int2", {q, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_col_only_int2", {{q, "q", 0}, {scale_col, "scale_col", 1}});

  int kElementsPerVector = 4;

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  auto x =
      torch::empty({rows, colsDst}, torch::dtype(torch::kBFloat16).device(q.device()));

  sym_dequant_col_only_int2_host(q.data_ptr<int8_t>(), (half_bf16 *)scale_col.data_ptr(),
                                 rows, colsSrc, colsDst, (half_bf16 *)x.data_ptr());

  return x;
}


torch::Tensor sym_dequantize_quantize(const torch::Tensor &q_in,
                                      const torch::Tensor &scale_row, const torch::Tensor &scale_col) {
  torch::checkAllContiguous("sym_dequantize_quantize", {{q_in, "q_in", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});
  torch::checkDeviceType("sym_dequantize_quantize", {q_in, scale_row, scale_col}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequantize_quantize", {{q_in, "q_in", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});
  uint32_t rowsDst = q_in.size(0);
  uint32_t colsSrc = q_in.size(1);

  uint32_t rowsSrc = rowsDst / kElementsPerVector;
  uint32_t colsDst = colsSrc * kElementsPerVector;

  auto q_out = torch::zeros({colsDst, rowsSrc}, torch::dtype(torch::kInt8).device(q_in.device()));

  sym_dequantize_quantize_host(q_in.data_ptr<int8_t>(), q_out.data_ptr<int8_t>(), (half_bf16 *)scale_row.data_ptr(), (half_bf16 *)scale_col.data_ptr(), rowsSrc, rowsDst, colsSrc, colsDst);

  return q_out;
}

torch::Tensor int4_to_int8(const torch::Tensor &q) {
  torch::checkAllContiguous("int4_to_int8", {{q, "q", 0}});
  torch::checkDeviceType("int4_to_int8", {q}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("int4_to_int8", {{q, "q", 0}});

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = q.size(1) * kElementsPerVector;

  auto q_out = torch::zeros({rows, colsDst}, torch::dtype(torch::kInt8).device(q.device()));

  int4_to_int8_host(q.data_ptr<int8_t>(), rows, colsSrc, colsDst, q_out.data_ptr<int8_t>());

  return q_out;
}


//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("matmul", &matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
  
  m.def("matmul_int8", &matmul_int8,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));

  m.def("sym_quant", &sym_quant,
        "input: (src: torch.Tensor(M x N, BF16, CUDA), scale: "
        "torch.Tensor(M x 1, BF16, CUDA))"
        "bits: int\n"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def("sym_quant_int8", &sym_quant_int8,
        "input: (src: torch.Tensor(M x N, BF16, CUDA), scale: "
        "torch.Tensor(M x 1, BF16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), INT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def("sym_quant_int2", &sym_quant_int2,
        "input: (src: torch.Tensor(M x N, BF16, CUDA), scale: "
        "torch.Tensor(M x 1, BF16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), INT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def ("sym_quant_int2_col", &sym_quant_int2_col,
        "input: (src: torch.Tensor(M x N, BF16, CUDA), scale: "
        "torch.Tensor(M x 1, BF16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), INT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def("sym_dequant", &sym_dequant,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "BF16), scale_col: torch.Tensor(1 x N, BF16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_row * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is BF16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("scale_col"),
        py::arg("bits"));

  m.def("sym_dequant_col_only", &sym_dequant_col_only,
        "input (x: torch.Tensor(M x N), scale_col: torch.Tensor(1 x N, BF16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is BF16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_col"), py::arg("bits"));
  
  m.def("sym_dequant_row_only", &sym_dequant_row_only,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, BF16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_row"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is BF16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("bits"));
    
  m.def("sym_dequant_row_only_int8", &sym_dequant_row_only_int8,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, BF16)"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_row",
        py::arg("q"), py::arg("scale_row"));
  
  m.def("sym_dequant_row_only_int2", &sym_dequant_row_only_int2,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, BF16)"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_row",
        py::arg("q"), py::arg("scale_row"));

  m.def("sym_dequant_col_only_int2", &sym_dequant_col_only_int2,
        "input (x: torch.Tensor(M x N), scale_col: torch.Tensor(1 x N, BF16)"
        "output: torch.Tensor(M x N, BF16)\n"
        "output = x * scale_col",
        py::arg("q"), py::arg("scale_col"));

  m.def("sym_dequantize_quantize", &sym_dequantize_quantize,
        "input (q_in: torch.Tensor(M x N, INT8), scale_row: torch.Tensor(M x 1, BF16), scale_col: torch.Tensor(1 x N, BF16)"
        "output: torch.Tensor(M x N, INT8)\n"
        "output = int4Packing(int4Rounding(int4Unpacking(q_in) * scale_row * scale_col)\n",
        py::arg("q_in"), py::arg("scale_row"), py::arg("scale_col"));

  m.def("int4_to_int8", &int4_to_int8,
        "input: (q: torch.Tensor(M x N, UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT8, CUDA)\n"
        "output = int4Unpacking(q)",
        py::arg("q"));
}
