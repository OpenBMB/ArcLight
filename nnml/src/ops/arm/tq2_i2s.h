#pragma once
#include "ops.h"   // block_tq2_0, QK_K, nnml_fp16_t, NNML_FP16_TO_FP32

// I2_S interleaved cache: a TQ2_0 weight matrix reordered into the 4-row-packed
// layout consumed by the I2_S NEON matmul kernel. All buffers are heap-allocated
// and owned by the cache (free with tq2_i2s_free_cache).
struct tq2_i2s_cache {
    uint8_t * packed;   // 4-row-packed 2-bit codes (n_groups * sub_blocks * 64 bytes)
    float   * scales;   // per I2S sub-block, 4 floats (one per row)
    int32_t * bsums;    // per I2S sub-block, 4 int32  (one per row)
    int      out_dim;
    int      in_dim;
};

// Packed byte count for the I2_S layout (0 on bad args).
size_t tq2_i2s_packed_size(int out_dim, int in_dim);

// Allocate + fill a cache (caller frees with tq2_i2s_free_cache).
// Returns nullptr on bad args / non-ARM.
tq2_i2s_cache * tq2_i2s_build_cache(const void * tq2_weight, int out_dim, int in_dim);

void tq2_i2s_free_cache(tq2_i2s_cache * c);

// Activation quant: f32 row -> int8 + scalar absmax scale + per-256 block bsums.
// Returns 0 on success, -1 on bad args / non-ARM.
int tq2_quantize_vec_i8(const float * vec, int n, int8_t * qvec, float * scale, int32_t * block_bsums);

// I2_S NEON matmul kernel. Computes output rows [row_begin, row_begin+row_count)
// of:  out[r] = dot(I2S_weight[r], activation).
//   c        : I2S cache (out_dim rows, in_dim). row_begin+row_count <= c->out_dim.
//   qvec     : int8 activation, in_dim elements (from tq2_quantize_vec_i8)
//   vec_scale: activation absmax scale (from tq2_quantize_vec_i8)
//   bsums    : per-256 activation block sums (from tq2_quantize_vec_i8), in_dim/256 ints
//   row_begin, row_count : output row range to compute
//   out      : output floats, row_count elements WRITTEN (out[0] = row row_begin, ...)
// No-op on non-ARM / bad args.
void tq2_matmul_i2s_neon(const tq2_i2s_cache * c, const int8_t * qvec, float vec_scale,
                         const int32_t * bsums, int row_begin, int row_count, float * out);
