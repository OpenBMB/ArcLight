// TQ2_0 I2_S data-prep primitives: reorder a TQ2_0 weight matrix into the
// 4-row-packed I2_S layout, and quantize an f32 activation row to int8.
//
// Ported from 158BitNet's bitnet_tq2_0_reorder_to_i2s and
// bitnet_tq2_0_quantize_vec_i8_impl (src/quant_tq2_0.c). The I2_S layout and
// bsums convention must match the reference exactly: Task 2's matmul kernel
// (ported from the same reference) depends on it.

#include "tq2_i2s.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <arm_neon.h>

// Local I2_S block size (4-row packed). Matches reference QK_I2S.
#define QK_I2S 64

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

// Extract a single 2-bit code from TQ2_0 interleaved layout.
// TQ2_0 layout: for byte at qs[g*32+m], bits[l*2+1:l*2] = code
//   for position g*128 + l*32 + m. Returns the code (0..3), value = code - 1.
static inline uint8_t tq2_0_extract_code(const uint8_t * qs, int pos) {
    int g = pos / 128;
    int l = (pos % 128) / 32;
    int m = pos % 32;
    return (qs[g * 32 + m] >> (l * 2)) & 3;
}

// Port of bitnet_tq2_0_i2s_packed_size.
size_t tq2_i2s_packed_size(int out_dim, int in_dim) {
    if (out_dim <= 0 || in_dim <= 0 || in_dim % QK_K != 0) {
        return 0;
    }
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;
    return (size_t)n_groups * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
}

// Port of bitnet_tq2_0_reorder_to_i2s, wrapped to allocate + return a cache.
tq2_i2s_cache * tq2_i2s_build_cache(const void * tq2_weight, int out_dim, int in_dim) {
    if (tq2_weight == nullptr || out_dim <= 0 || in_dim <= 0 || in_dim % QK_K != 0) {
        return nullptr;
    }

    const uint8_t * bytes = (const uint8_t *)tq2_weight;
    int blocks_per_row = in_dim / QK_K;
    if (blocks_per_row <= 0) return nullptr;

    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;

    size_t packed_size  = tq2_i2s_packed_size(out_dim, in_dim);
    size_t scales_count = (size_t)n_groups * (size_t)blocks_per_row * 4u;
    size_t bsums_count  = (size_t)n_groups * (size_t)sub_blocks_per_group * 4u;

    tq2_i2s_cache * c = (tq2_i2s_cache *)std::malloc(sizeof(tq2_i2s_cache));
    if (c == nullptr) return nullptr;
    c->packed  = (uint8_t *)std::malloc(packed_size);
    c->scales  = (float *)  std::malloc(scales_count * sizeof(float));
    c->bsums   = (int32_t *)std::malloc(bsums_count  * sizeof(int32_t));
    c->out_dim = out_dim;
    c->in_dim  = in_dim;
    if (c->packed == nullptr || c->scales == nullptr || c->bsums == nullptr) {
        std::free(c->packed);
        std::free(c->scales);
        std::free(c->bsums);
        std::free(c);
        return nullptr;
    }

    uint8_t * packed       = c->packed;
    float   * packed_scales = c->scales;
    int32_t * packed_bsums  = c->bsums;

    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;

        for (int sb = 0; sb < sub_blocks_per_group; ++sb) {
            int elem_base = sb * QK_I2S;

            // Packed data: QK_I2S bytes for this sub-block.
            uint8_t * dst = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S
                         + (size_t)sb * (size_t)QK_I2S;
            // Scales: 4 floats per TQ2_0 block (one per row).
            float * dst_scales = packed_scales + ((size_t)grp * (size_t)blocks_per_row +
                                                  (size_t)(elem_base / QK_K)) * 4u;
            // Bsums: 4 int32 for this sub-block (one per row).
            int32_t * dst_bsums = packed_bsums + ((size_t)grp * (size_t)sub_blocks_per_group +
                                                  (size_t)sb) * 4u;

            // Clear destination bytes (for padding rows).
            std::memset(dst, 0, QK_I2S);

            for (int r = 0; r < 4; ++r) {
                int row = row_base + r;
                float row_scale = 0.0f;
                int32_t row_bsum = 0;

                if (row < out_dim) {
                    size_t row_offset = (size_t)row * (size_t)blocks_per_row * sizeof(block_tq2_0);

                    // Determine which TQ2_0 block this sub-block starts in.
                    int tq2_blk = elem_base / QK_K;
                    const block_tq2_0 * block =
                        (const block_tq2_0 *)(bytes + row_offset +
                            (size_t)tq2_blk * sizeof(block_tq2_0));
                    row_scale = NNML_FP16_TO_FP32(block->d);

                    // Decode all 64 codes for this row in this sub-block and pack.
                    int32_t code_sum = 0;
                    for (int e = 0; e < QK_I2S; ++e) {
                        int global_pos  = elem_base + e;
                        int blk         = global_pos / QK_K;
                        int pos_in_blk  = global_pos % QK_K;
                        const block_tq2_0 * blk_ptr =
                            (const block_tq2_0 *)(bytes + row_offset +
                                (size_t)blk * sizeof(block_tq2_0));
                        uint8_t code = tq2_0_extract_code(blk_ptr->qs, pos_in_blk);
                        // Pack: row r's code goes into bits [(3-r)*2+1 : (3-r)*2].
                        dst[e] |= (uint8_t)(code << ((3 - r) * 2));
                        code_sum += (int32_t)code;
                    }
                    // bsum = sum(code - 1) = sum(code) - QK_I2S.
                    row_bsum = code_sum - QK_I2S;
                }

                dst_scales[r] = row_scale;
                dst_bsums[r]  = row_bsum;
            }
        }
    }

    return c;
}

void tq2_i2s_free_cache(tq2_i2s_cache * c) {
    if (c == nullptr) return;
    std::free(c->packed);
    std::free(c->scales);
    std::free(c->bsums);
    std::free(c);
}

// Port of bitnet_tq2_0_quantize_vec_i8_impl (NEON path).
int tq2_quantize_vec_i8(const float * vec, int n, int8_t * qvec, float * scale, int32_t * block_bsums) {
    float max_abs = 0.0f;
    int bsums_computed = 0;

    if (vec == nullptr || qvec == nullptr || scale == nullptr || n <= 0) {
        return -1;
    }

    // NEON-accelerated max_abs.
    {
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(vec + i);
            max_vec = vmaxq_f32(max_vec, vabsq_f32(v));
        }
        max_abs = vmaxvq_f32(max_vec);
        for (; i < n; ++i) {
            float a = fabsf(vec[i]);
            if (a > max_abs) max_abs = a;
        }
    }

    if (max_abs <= 0.0f) {
        std::memset(qvec, 0, (size_t)n * sizeof(*qvec));
        *scale = 1.0f;
        if (block_bsums != nullptr) {
            int n_blocks = n / QK_K;
            std::memset(block_bsums, 0, (size_t)n_blocks * sizeof(*block_bsums));
        }
        return 0;
    }

    *scale = max_abs / 127.0f;
    const float inv_scale = 127.0f / max_abs;

    // NEON-accelerated quantize: float -> int8 with clamping.
    {
        float32x4_t inv_s = vdupq_n_f32(inv_scale);
        float32x4_t lo = vdupq_n_f32(-127.0f);
        float32x4_t hi = vdupq_n_f32(127.0f);
        int n_blocks = n / QK_K;
        int i = 0;
        if (block_bsums != nullptr && n % QK_K == 0) {
            for (int block = 0; block < n_blocks; ++block) {
                int16x8_t bsum16 = vdupq_n_s16(0);
                const float * src = vec + (size_t)block * QK_K;
                int8_t * dst = qvec + (size_t)block * QK_K;

                for (int j = 0; j < QK_K; j += 16) {
                    float32x4_t v0 = vmulq_f32(vld1q_f32(src + j + 0), inv_s);
                    float32x4_t v1 = vmulq_f32(vld1q_f32(src + j + 4), inv_s);
                    float32x4_t v2 = vmulq_f32(vld1q_f32(src + j + 8), inv_s);
                    float32x4_t v3 = vmulq_f32(vld1q_f32(src + j + 12), inv_s);
                    int16x8_t lo16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                    int16x8_t hi16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                    int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                    vst1q_s8(dst + j, i8);
                    bsum16 = vaddq_s16(bsum16, vpaddlq_s8(i8));
                }
                block_bsums[block] = (int32_t)vaddlvq_s16(bsum16);
            }
            i = n_blocks * QK_K;
        } else {
            if (block_bsums != nullptr) {
                std::memset(block_bsums, 0, (size_t)n_blocks * sizeof(*block_bsums));
            }
            for (; i + 15 < n; i += 16) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(vec + i + 0), inv_s);
                float32x4_t v1 = vmulq_f32(vld1q_f32(vec + i + 4), inv_s);
                float32x4_t v2 = vmulq_f32(vld1q_f32(vec + i + 8), inv_s);
                float32x4_t v3 = vmulq_f32(vld1q_f32(vec + i + 12), inv_s);
                int16x8_t lo16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                int16x8_t hi16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                vst1q_s8(qvec + i, i8);
                if (block_bsums != nullptr) {
                    int block = i / QK_K;
                    if (block < n_blocks) {
                        block_bsums[block] += (int32_t)vaddlvq_s8(i8);
                    }
                }
            }
        }
        for (; i < n; ++i) {
            int q = (int)lrintf(vec[i] * inv_scale);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            qvec[i] = (int8_t)q;
            if (block_bsums != nullptr) {
                int block = i / QK_K;
                if (block < n_blocks) {
                    block_bsums[block] += q;
                }
            }
        }
        bsums_computed = (block_bsums != nullptr);
    }

    // Compute per-block bsums using NEON (fallback path).
    if (block_bsums != nullptr && !bsums_computed) {
        int n_blocks = n / QK_K;
        for (int k = 0; k < n_blocks; ++k) {
            const int8_t * qv = qvec + (size_t)k * QK_K;
            int32_t bsum = 0;
            for (int i = 0; i < QK_K; i += 16) {
                bsum += (int32_t)vaddlvq_s8(vld1q_s8(qv + i));
            }
            block_bsums[k] = bsum;
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// I2_S NEON matmul kernel. Ports bitnet_tq2_0_matmul_i2s_neon +
// i2s_matmul_4rows_neon from 158BitNet/src/quant_tq2_0.c.
//
// Cache layout (written by tq2_i2s_build_cache) — MUST match the reads below:
//   packed [grp*sbpg*QK_I2S + sb*QK_I2S + e] : byte e of I2S sub-block sb of
//                                               group grp (4 rows interleaved).
//   scales[(grp*blocks_per_row + blk)*4 + r] : per-row weight scale (one per
//                                               QK_K block; shared by its 4
//                                               I2S sub-blocks).
//   The activation bsums (parameter) are per QK_K block: bsums[blk].
// ---------------------------------------------------------------------------

// Lookup tables for vqtbl1q: extract the 4 per-row 2-bit codes from each packed
// byte. The high nibble holds rows 0/1 (codes in bits [7:6]/[5:4]); the low
// nibble holds rows 2/3 (bits [3:2]/[1:0]). Unsigned variant emits raw codes
// 0..3 (caller applies the activation bsum correction); signed variant emits
// code-1 directly (no correction). Faithful port of the reference LUTs.
static const uint8_t i2s_lut_hi2_unsigned_data[16] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};
static const uint8_t i2s_lut_lo2_unsigned_data[16] = {
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
};
static const uint8_t i2s_lut_hi2_signed_data[16] = {
    255, 255, 255, 255, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2
};
static const uint8_t i2s_lut_lo2_signed_data[16] = {
    255, 0, 1, 2, 255, 0, 1, 2, 255, 0, 1, 2, 255, 0, 1, 2
};

#define I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3, mask_0f, lut_hi2, lut_lo2) do { \
        uint8x16_t i2s_hi_nib = vshrq_n_u8((pk), 4);         \
        uint8x16_t i2s_lo_nib = vandq_u8((pk), (mask_0f));   \
        (c0) = vreinterpretq_s8_u8(vqtbl1q_u8((lut_hi2), i2s_hi_nib)); \
        (c1) = vreinterpretq_s8_u8(vqtbl1q_u8((lut_lo2), i2s_hi_nib)); \
        (c2) = vreinterpretq_s8_u8(vqtbl1q_u8((lut_hi2), i2s_lo_nib)); \
        (c3) = vreinterpretq_s8_u8(vqtbl1q_u8((lut_lo2), i2s_lo_nib)); \
    } while (0)

// Dot one 4-row group against the int8 activation. Ports
// i2s_matmul_4rows_neon (unsigned-LUT + activation-bsum path) faithfully,
// including the dual-accumulator unroll.
static inline void tq2_i2s_matmul_4rows_neon(
    const uint8_t * packed_grp, const float * scales_grp,
    int blocks_per_row, const int8_t * qvec,
    const int32_t * bsums, float vec_scale,
    float * out_0, float * out_1, float * out_2, float * out_3) {

    const int use_signed_lut = (bsums == nullptr);
    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    const uint8x16_t lut_hi2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_hi2_signed_data :
                                        i2s_lut_hi2_unsigned_data);
    const uint8x16_t lut_lo2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_lo2_signed_data :
                                        i2s_lut_lo2_unsigned_data);

    // Process per TQ2_0 block (each = 4 I2S sub-blocks of QK_I2S=64 bytes).
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        // Dual accumulators to break the vdotq dependency chain.
        int32x4_t acc_0_lo = vdupq_n_s32(0), acc_0_hi = vdupq_n_s32(0);
        int32x4_t acc_1_lo = vdupq_n_s32(0), acc_1_hi = vdupq_n_s32(0);
        int32x4_t acc_2_lo = vdupq_n_s32(0), acc_2_hi = vdupq_n_s32(0);
        int32x4_t acc_3_lo = vdupq_n_s32(0), acc_3_hi = vdupq_n_s32(0);

        for (int sub = 0; sub < 4; ++sub) {
            int sb = blk * 4 + sub;
            const uint8_t * pb = packed_grp + (size_t)sb * (size_t)QK_I2S;
            const int8_t  * qv = qvec       + (size_t)sb * (size_t)QK_I2S;

            for (int i = 0; i < QK_I2S; i += 32) {
                // First 16 bytes.
                {
                    uint8x16_t pk = vld1q_u8(pb + i);
                    int8x16_t  v  = vld1q_s8(qv + i);
                    int8x16_t c0, c1, c2, c3;
                    I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3, mask_0f, lut_hi2, lut_lo2);

                    acc_0_lo = vdotq_s32(acc_0_lo, c0, v);
                    acc_1_lo = vdotq_s32(acc_1_lo, c1, v);
                    acc_2_lo = vdotq_s32(acc_2_lo, c2, v);
                    acc_3_lo = vdotq_s32(acc_3_lo, c3, v);
                }
                // Second 16 bytes.
                {
                    uint8x16_t pk = vld1q_u8(pb + i + 16);
                    int8x16_t  v  = vld1q_s8(qv + i + 16);
                    int8x16_t c0, c1, c2, c3;
                    I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3, mask_0f, lut_hi2, lut_lo2);

                    acc_0_hi = vdotq_s32(acc_0_hi, c0, v);
                    acc_1_hi = vdotq_s32(acc_1_hi, c1, v);
                    acc_2_hi = vdotq_s32(acc_2_hi, c2, v);
                    acc_3_hi = vdotq_s32(acc_3_hi, c3, v);
                }
            }
        }

        // Per-row weight scale for this TQ2_0 block (same across its 4 sub-blocks).
        const float * sb_scales = scales_grp + (size_t)blk * 4u;
        float s0 = sb_scales[0], s1 = sb_scales[1], s2 = sb_scales[2], s3 = sb_scales[3];

        int32_t dot_0 = vaddvq_s32(vaddq_s32(acc_0_lo, acc_0_hi));
        int32_t dot_1 = vaddvq_s32(vaddq_s32(acc_1_lo, acc_1_hi));
        int32_t dot_2 = vaddvq_s32(vaddq_s32(acc_2_lo, acc_2_hi));
        int32_t dot_3 = vaddvq_s32(vaddq_s32(acc_3_lo, acc_3_hi));

        // Unsigned-LUT path emits raw codes 0..3; subtract the activation block
        // sum to recover the code-1 (ternary) dot product.
        if (!use_signed_lut) {
            const int32_t bsum = bsums[blk];
            dot_0 -= bsum;
            dot_1 -= bsum;
            dot_2 -= bsum;
            dot_3 -= bsum;
        }

        sum_0 += (float)dot_0 * s0;
        sum_1 += (float)dot_1 * s1;
        sum_2 += (float)dot_2 * s2;
        sum_3 += (float)dot_3 * s3;
    }

    *out_0 = sum_0 * vec_scale;
    *out_1 = sum_1 * vec_scale;
    *out_2 = sum_2 * vec_scale;
    *out_3 = sum_3 * vec_scale;
}

void tq2_matmul_i2s_neon(const tq2_i2s_cache * c, const int8_t * qvec, float vec_scale,
                         const int32_t * bsums, int row_begin, int row_count, float * out) {
    if (c == nullptr || c->packed == nullptr || c->scales == nullptr ||
        qvec == nullptr || out == nullptr || row_begin < 0 || row_count <= 0) {
        return;
    }
    if (row_begin >= c->out_dim) return;

    int in_dim  = c->in_dim;
    int out_dim = c->out_dim;
    int blocks_per_row = in_dim / QK_K;
    if (blocks_per_row <= 0) return;

    // Clamp the requested range to the real output rows.
    int row_end = row_begin + row_count;
    if (row_end > out_dim) row_end = out_dim;
    if (row_begin >= row_end) return;

    int out_dim_padded     = (out_dim + 3) & ~3;
    int n_groups           = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;

    // Iterate over the 4-row groups that overlap [row_begin, row_end). Each
    // group shares 64 bytes of packed data across its 4 rows, so the group is
    // the unit of work; boundary rows outside the request are computed but
    // discarded. Group-aligned splits (Task 3) avoid the discard.
    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        if (row_base + 4 <= row_begin) continue;   // group entirely below the range
        if (row_base >= row_end) break;            // group entirely above the range

        const uint8_t * packed_grp = c->packed  + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float   * scales_grp = c->scales  + (size_t)grp * (size_t)blocks_per_row * 4u;

        float r0, r1, r2, r3;
        tq2_i2s_matmul_4rows_neon(packed_grp, scales_grp, blocks_per_row,
                                  qvec, bsums, vec_scale, &r0, &r1, &r2, &r3);

        float rr[4] = { r0, r1, r2, r3 };
        for (int k = 0; k < 4; ++k) {
            int row = row_base + k;
            if (row >= row_begin && row < row_end) {
                out[row - row_begin] = rr[k];
            }
        }
    }
}

#else  // !__ARM_NEON || !__ARM_FEATURE_DOTPROD

// Non-ARM stubs so the TU still links (dispatch never calls them here).
size_t tq2_i2s_packed_size(int /*out_dim*/, int /*in_dim*/) { return 0; }

tq2_i2s_cache * tq2_i2s_build_cache(const void * /*tq2_weight*/, int /*out_dim*/, int /*in_dim*/) {
    return nullptr;
}

void tq2_i2s_free_cache(tq2_i2s_cache * c) { (void)c; }

int tq2_quantize_vec_i8(const float * /*vec*/, int /*n*/, int8_t * /*qvec*/,
                        float * /*scale*/, int32_t * /*block_bsums*/) {
    return -1;
}

void tq2_matmul_i2s_neon(const tq2_i2s_cache * /*c*/, const int8_t * /*qvec*/, float /*vec_scale*/,
                         const int32_t * /*bsums*/, int /*row_begin*/, int /*row_count*/,
                         float * /*out*/) {
    // No-op on non-ARM (dispatch never calls here).
}

#endif  // defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
