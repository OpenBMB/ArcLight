#include <cassert>

#include "ops.h"
#include "tensor.h"

nnml_type_traits place_holder = {
    .type_name                = nullptr,
    .blck_size                = -1,
    .blck_size_interleave     = -1,
    .type_size                = 0,
    .is_quantized             = false,
    .to_float                 = nullptr,
    .from_float_ref           = nullptr,
    .from_float_cpu           = nullptr,
    .vec_dot                  = nullptr,
    .vec_dot_type             = NNML_TYPE_COUNT,
    .nrows                    = -1,
};

const nnml_type_traits type_traits[NNML_TYPE_COUNT] = {
    [NNML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .from_float_cpu           = (nnml_from_float_t) nnml_cpu_fp32_to_fp32,
        .vec_dot                  = (nnml_vec_dot_t) nnml_vec_dot_f32,
        .vec_dot_type             = NNML_TYPE_F32,
        .nrows                    = 1,
    },
    [NNML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(nnml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (nnml_to_float_t) nnml_fp16_to_fp32_row,
        .from_float_ref           = (nnml_from_float_t) nnml_fp32_to_fp16_row,
        .from_float_cpu           = (nnml_from_float_t) nnml_cpu_fp32_to_fp16,
        .vec_dot                  = (nnml_vec_dot_t) nnml_vec_dot_f16,
        .vec_dot_type             = NNML_TYPE_F16,
        .nrows                    = 1,
    },
    [NNML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (nnml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (nnml_from_float_t) quantize_row_q4_0_ref,
        .from_float_cpu           = quantize_row_q4_0,
        .vec_dot                  = nnml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = NNML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [NNML_TYPE_PH_3] = place_holder,
    [NNML_TYPE_PH_4] = place_holder,
    [NNML_TYPE_PH_5] = place_holder,
    [NNML_TYPE_PH_6] = place_holder,
    [NNML_TYPE_PH_7] = place_holder,
    [NNML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (nnml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (nnml_from_float_t) quantize_row_q8_0_ref,
        .from_float_cpu           = quantize_row_q8_0,
        .vec_dot                  = nnml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = NNML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [NNML_TYPE_PH_9] = place_holder,
    [NNML_TYPE_PH_10] = place_holder,
    [NNML_TYPE_PH_11] = place_holder,
    [NNML_TYPE_PH_12] = place_holder,
    [NNML_TYPE_PH_13] = place_holder,
    [NNML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (nnml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (nnml_from_float_t) quantize_row_q6_K_ref,
        .vec_dot                  = nnml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = NNML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [NNML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
        .from_float_cpu           = quantize_row_q8_K,
    },
    [NNML_TYPE_PH_16] = place_holder,
    [NNML_TYPE_PH_17] = place_holder,
    [NNML_TYPE_PH_18] = place_holder,
    [NNML_TYPE_PH_19] = place_holder,
    [NNML_TYPE_PH_20] = place_holder,
    [NNML_TYPE_PH_21] = place_holder,
    [NNML_TYPE_PH_22] = place_holder,
    [NNML_TYPE_PH_23] = place_holder,
    [NNML_TYPE_PH_24] = place_holder,
    [NNML_TYPE_PH_25] = place_holder,
    [NNML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [NNML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = sizeof(int64_t),
        .is_quantized             = false,
    },
    [NNML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = sizeof(double),
        .is_quantized             = false,
    },
};

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;
    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }
    const int end = QK4_0 * 2 / blck_size_interleave;
    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;
            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;
            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        NNML_ASSERT(false);
    }
    return out;
}

static int repack_q4_0_to_q4_0_4_bl(struct nnml_tensor * t, int interleave_block, const void * NNML_RESTRICT data, size_t data_size) {
    NNML_ASSERT(t->get_data_type() == NNML_TYPE_Q4_0);
    NNML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->tensor_data();
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nrow = t->n_rows();
    int nblocks = t->get_elements(0) / QK4_0;

    NNML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->get_elements(1) % nrows_interleaved != 0 || t->get_elements(0) % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    NNML_UNUSED(data_size);
}

template <> int repack<block_q4_0, 4, 4>(nnml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q4_0, 8, 4>(nnml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_0, 8, 8>(nnml_tensor * t, const void * data, size_t data_size) {
    // return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
    abort(); // to be implemented
}

int format_repack(nnml_tensor * weight_tensor) {
    if (weight_tensor->get_data_type() != NNML_TYPE_Q4_0) {
        return 0; // only weight tensors are supported
    } else {
        size_t data_size = nnml_nbytes(weight_tensor);
        void * ptr = malloc(data_size);
        if (!ptr) {
            return -1; // malloc failed
        }
        memcpy(ptr, weight_tensor->tensor_data(), data_size);
        if (nnml_cpu_has_avx2() || (nnml_cpu_has_sve() && nnml_cpu_has_matmul_int8() && nnml_cpu_get_sve_cnt() == QK8_0)) {
            if (weight_tensor->get_elements(1) % 8 == 0) {
                repack<block_q4_0, 8, 8>(weight_tensor, ptr, data_size);
            }
        }
        if (nnml_cpu_has_neon() && nnml_cpu_has_matmul_int8()) {
            if (weight_tensor->get_elements(1) % 4 == 0) {
                repack<block_q4_0, 8, 4>(weight_tensor, ptr, data_size);
            }
        }
        if (nnml_cpu_has_neon() && nnml_cpu_has_dotprod()) {
            if (weight_tensor->get_elements(1) % 4 == 0) {
                repack<block_q4_0, 4, 4>(weight_tensor, ptr, data_size);
            }
        }
        free(ptr);
        return 0;
    }
}

// Helpers

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * NNML_RESTRICT x, int8_t * NNML_RESTRICT L, int rmse_type,
        const float * NNML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUNNY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

// F16

void nnml_fp16_to_fp32_row(const nnml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = NNML_FP16_TO_FP32(x[i]);
    }
}

void nnml_fp32_to_fp16_row(const float * x, nnml_fp16_t * y, int64_t n) {
    int i = 0;
    for (; i < n; ++i) {
        y[i] = NNML_FP32_TO_FP16(x[i]);
    }
}

// Q4_0

void dequantize_row_q4_0(const block_q4_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = NNML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void quantize_row_q4_0_ref(const float * NNML_RESTRICT x, block_q4_0 * NNML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }
        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = NNML_FP32_TO_FP16(d);
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;
            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));
            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

// Q6_K

void dequantize_row_q6_K(const block_q6_K * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = NNML_FP16_TO_FP32(x[i].d);

        const uint8_t * NNML_RESTRICT ql = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const int8_t  * NNML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void quantize_row_q6_K_ref(const float * NNML_RESTRICT x, block_q6_K * NNML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = NNML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = NNML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = NNML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * NNML_RESTRICT ql = y[i].ql;
        uint8_t * NNML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

// Q8_0

void dequantize_row_q8_0(const block_q8_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = NNML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void quantize_row_q8_0_ref(const float * NNML_RESTRICT x, block_q8_0 * NNML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = NNML_FP32_TO_FP16(d);
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

// Q8_K

void quantize_row_q8_K_ref(const float * NNML_RESTRICT x, block_q8_K * NNML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void quantize_row_q8_K(const float * NNML_RESTRICT x, void * NNML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, (block_q8_K * NNML_RESTRICT)y, k);
}

int64_t nnml_blck_size(nnml_type type) {
    return type_traits[type].blck_size;
}

size_t nnml_type_size(nnml_type type) {
    // printf("type: %d\n", type);
    return type_traits[type].type_size;
}

const char * nnml_type_name(nnml_type type) {
    return type < NNML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

bool nnml_is_quantized(nnml_type type) {
    return type_traits[type].is_quantized;
}

const nnml_type_traits * nnml_get_type_traits(nnml_type type) {
    NNML_ASSERT(type < NNML_TYPE_COUNT);
    return &type_traits[type];
}
