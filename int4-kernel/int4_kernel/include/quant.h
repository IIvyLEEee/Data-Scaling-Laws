#pragma once

#include <common.h>

typedef __nv_bfloat16 half_bf16;

void sym_quant_host(
    const half_bf16 *x,
    const half_bf16 *scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);

void sym_quant_int2_host(
    const half_bf16 *x,
    const half_bf16 *scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    int8_t *q
);

void sym_quant_int8_host(
    const half_bf16 *x,
    const half_bf16 *scale,
    uint32_t rows,
    uint32_t cols,
    int8_t *q
);

void sym_quant_int2_col_host(
    const half_bf16 *x,
    const half_bf16 *scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    int8_t *q
);

void sym_dequant_host(
    const int32_t *q,
    const half_bf16 *scale_row,
    const half_bf16 *scale_col,
    uint32_t rows,
    uint32_t cols,
    half_bf16 *x
);

void sym_dequant_col_only_host(
    const int8_t *q,
    const half_bf16 *scale_col,
    uint32_t rowsSrc,
    uint32_t rowsDst,
    uint32_t cols,
    half_bf16 *x
);

void sym_dequant_row_only_host(
    const int8_t *q,
    const half_bf16 *scale_row,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    half_bf16 *x
);

void sym_dequant_row_only_int8_host(
    const int8_t *q,
    const half_bf16 *scale_row,
    uint32_t rows,
    uint32_t cols,
    half_bf16 *x
);

void sym_dequant_row_only_int2_host(
    const int8_t *q,
    const half_bf16 *scale_row,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    half_bf16 *x
);

void sym_dequant_col_only_int2_host(
    const int8_t *q,
    const half_bf16 *scale_col,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    half_bf16 *x
);

void sym_dequantize_quantize_host(
    const int8_t *q_in, 
    int8_t *q_out,
    const half_bf16 *scale_row, 
    const half_bf16 *scale_col, 
    uint32_t rowsSrc, 
    uint32_t rowsDst, 
    uint32_t colsSrc, 
    uint32_t colsDst
);

void int4_to_int8_host(
    const int8_t *q,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    int8_t *q_out
);