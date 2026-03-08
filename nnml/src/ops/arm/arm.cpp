#include <cassert>

#include <cmath>
#include <arm_neon.h>
#include <unistd.h>

#include "ops.h"
#include "tensor.h"

#define nnml_int8x16x2_t  int8x16x2_t
#define nnml_int8x16x4_t  int8x16x4_t
#define nnml_int16x8x2_t  int16x8x2_t
#define nnml_uint8x16x4_t uint8x16x4_t
#define nnml_uint8x16x2_t uint8x16x2_t

#define nnml_vld1q_s16_x2 vld1q_s16_x2
#define nnml_vld1q_u8_x2  vld1q_u8_x2
#define nnml_vld1q_u8_x4  vld1q_u8_x4
#define nnml_vld1q_s8_x2  vld1q_s8_x2
#define nnml_vld1q_s8_x4  vld1q_s8_x4
#define nnml_vqtbl1q_s8   vqtbl1q_s8
#define nnml_vqtbl1q_u8   vqtbl1q_u8

#if defined(__ARM_NEON)
    #define NNML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
    #define NNML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)

    #define NNML_CPU_FP16_TO_FP32(x) NNML_CPU_COMPUTE_FP16_TO_FP32(x)

    static inline float neon_compute_fp16_to_fp32(nnml_fp16_t h) {
        __fp16 tmp;
        memcpy(&tmp, &h, sizeof(nnml_fp16_t));
        return (float)tmp;
    }

    static inline nnml_fp16_t neon_compute_fp32_to_fp16(float f) {
        nnml_fp16_t res;
        __fp16 tmp = f;
        memcpy(&res, &tmp, sizeof(nnml_fp16_t));
        return res;
    }
#endif

#if !defined(NNML_CPU_FP32_TO_FP16)
#define NNML_CPU_FP32_TO_FP16(x) NNML_COMPUTE_FP32_TO_FP16(x)
#endif


#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)
    #define NNML_F32_VEC_LOAD   NNML_F32xt_LOAD
    #define NNML_F32_VEC_FMA    NNML_F32xt_FMA
    #define NNML_F32_VEC_REDUCE NNML_F32xt_REDUCE
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
    #define NNML_F32_STEP 16
    #define NNML_F32_EPR  4
    #define NNML_F32x4          float32x4_t
    #define NNML_F32_VEC        NNML_F32x4
    #define NNML_F32x4_ZERO     vdupq_n_f32(0.0f)
    #define NNML_F32_VEC_ZERO   NNML_F32x4_ZERO
    #define NNML_F32x4_LOAD     vld1q_f32
    #define NNML_F32_VEC_LOAD   NNML_F32x4_LOAD
    #define NNML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define NNML_F32_VEC_FMA    NNML_F32x4_FMA
    #define NNML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
    #define NNML_F32x4_REDUCE(res, x)                       \
    {                                                       \
        int offset = NNML_F32_ARR >> 1;                     \
        for (int i = 0; i < offset; ++i) {                  \
            (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
        }                                                   \
        offset >>= 1;                                       \
        for (int i = 0; i < offset; ++i) {                  \
            (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
        }                                                   \
        offset >>= 1;                                       \
        for (int i = 0; i < offset; ++i) {                  \
            (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
        }                                                   \
        (res) = (nnml_float) NNML_F32x4_REDUCE_ONE((x)[0]); \
    }
    #define NNML_F32_VEC_REDUCE NNML_F32x4_REDUCE
    #define NNML_F32x4_SET1(x)  vdupq_n_f32(x)
    #define NNML_F32_VEC_SET1   NNML_F32x4_SET1
    #define NNML_F32x4_MUL      vmulq_f32
    #define NNML_F32_VEC_MUL    NNML_F32x4_MUL
    #define NNML_F32x4_STORE    vst1q_f32
    #define NNML_F32_VEC_STORE  NNML_F32x4_STORE

    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        #define NNML_F16_STEP 32
        #define NNML_F16_EPR  8
        #define NNML_F16x8              float16x8_t
        #define NNML_F16_VEC            NNML_F16x8
        #define NNML_F16x8_MUL          vmulq_f16
        #define NNML_F16_VEC_MUL        NNML_F16x8_MUL
        #define NNML_F16x8_SET1(x)      vdupq_n_f16(x)
        #define NNML_F16_VEC_SET1       NNML_F16x8_SET1
        #define NNML_F16x8_ZERO         vdupq_n_f16(0.0f)
        #define NNML_F16_VEC_ZERO       NNML_F16x8_ZERO
        #define NNML_F16x8_LOAD(x)      vld1q_f16((const __fp16 *)(x))
        #define NNML_F16_VEC_LOAD(p, i) NNML_F16x8_LOAD(p)
        #define NNML_F16x8_STORE        vst1q_f16
        #define NNML_F16_VEC_STORE(p, r, i) NNML_F16x8_STORE((__fp16 *)(p), (r)[i])
        #define NNML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
        #define NNML_F16_VEC_FMA        NNML_F16x8_FMA
        #define NNML_F16x8_REDUCE(res, x)                               \
        do {                                                            \
            int offset = NNML_F16_ARR >> 1;                             \
            for (int i = 0; i < offset; ++i) {                          \
                (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
            }                                                           \
            offset >>= 1;                                               \
            for (int i = 0; i < offset; ++i) {                          \
                (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
            }                                                           \
            offset >>= 1;                                               \
            for (int i = 0; i < offset; ++i) {                          \
                (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
            }                                                           \
            const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 ((x)[0])); \
            const float32x4_t t1 = vcvt_f32_f16(vget_high_f16((x)[0])); \
            (res) = (nnml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
        } while (0)
        #define NNML_F16_VEC_REDUCE         NNML_F16x8_REDUCE
    #else
        #define NNML_F16_STEP 16
        #define NNML_F16_EPR  4
        #define NNML_F32Cx4              float32x4_t
        #define NNML_F16_VEC             NNML_F32Cx4
        #define NNML_F32Cx4_MUL          vmulq_f32
        #define NNML_F16_VEC_MUL         NNML_F32Cx4_MUL
        #define NNML_F32Cx4_SET1(x)      vdupq_n_f32(x)
        #define NNML_F16_VEC_SET1        NNML_F32Cx4_SET1
        #define NNML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
        #define NNML_F16_VEC_ZERO        NNML_F32Cx4_ZERO
        #define NNML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const __fp16 *)(x)))
        #define NNML_F16_VEC_LOAD(p, i)  NNML_F32Cx4_LOAD(p)
        #define NNML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
        #define NNML_F16_VEC_STORE(p, r, i) NNML_F32Cx4_STORE((__fp16 *)(p), r[i])
        #define NNML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
        #define NNML_F16_VEC_FMA         NNML_F32Cx4_FMA
        #define NNML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
        #define NNML_F32x4_REDUCE(res, x)                       \
        {                                                       \
            int offset = NNML_F32_ARR >> 1;                     \
            for (int i = 0; i < offset; ++i) {                  \
                (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
            }                                                   \
            offset >>= 1;                                       \
            for (int i = 0; i < offset; ++i) {                  \
                (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
            }                                                   \
            offset >>= 1;                                       \
            for (int i = 0; i < offset; ++i) {                  \
                (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
            }                                                   \
            (res) = (nnml_float) NNML_F32x4_REDUCE_ONE((x)[0]); \
        }
        #define NNML_F32Cx4_REDUCE       NNML_F32x4_REDUCE
        #define NNML_F16_VEC_REDUCE      NNML_F32Cx4_REDUCE
    #endif
#else
    ERROR "No SIMD support found"
#endif

#define NNML_F32_ARR (NNML_F32_STEP/NNML_F32_EPR)
#define NNML_F16_ARR (NNML_F16_STEP/NNML_F16_EPR)

void nnml_cpu_fp16_to_fp32(const nnml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;

    for (; i < n; ++i) {
        y[i] = NNML_CPU_FP16_TO_FP32(x[i]);
    }
}

float f16_to_f32(nnml_fp16_t x) {
    return NNML_CPU_FP16_TO_FP32(x);
}

void nnml_vec_dot_f32(int n, float * NNML_RESTRICT s, size_t bs, const float * NNML_RESTRICT x, size_t bx, const float * NNML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    float sumf = 0.0f;

    #if defined(__ARM_FEATURE_SVE)
        // now this part has no implementation
        const int sve_register_length = nnml_cpu_get_sve_cnt() * 8;
        const int nnml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int nnml_f32_step = 8 * nnml_f32_epr; // choose 8 SVE registers

        const int np = (n & ~(nnml_f32_step - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        for (int i = 0; i < np; i += nnml_f32_step) {
            ax1 = NNML_F32_VEC_LOAD(x + i);
            ay1 = NNML_F32_VEC_LOAD(y + i);
            sum1 = NNML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = NNML_F32_VEC_LOAD(x + i + 1*nnml_f32_epr);
            ay2 = NNML_F32_VEC_LOAD(y + i + 1*nnml_f32_epr);
            sum2 = NNML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = NNML_F32_VEC_LOAD(x + i + 2*nnml_f32_epr);
            ay3 = NNML_F32_VEC_LOAD(y + i + 2*nnml_f32_epr);
            sum3 = NNML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = NNML_F32_VEC_LOAD(x + i + 3*nnml_f32_epr);
            ay4 = NNML_F32_VEC_LOAD(y + i + 3*nnml_f32_epr);
            sum4 = NNML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = NNML_F32_VEC_LOAD(x + i + 4*nnml_f32_epr);
            ay5 = NNML_F32_VEC_LOAD(y + i + 4*nnml_f32_epr);
            sum5 = NNML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = NNML_F32_VEC_LOAD(x + i + 5*nnml_f32_epr);
            ay6 = NNML_F32_VEC_LOAD(y + i + 5*nnml_f32_epr);
            sum6 = NNML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = NNML_F32_VEC_LOAD(x + i + 6*nnml_f32_epr);
            ay7 = NNML_F32_VEC_LOAD(y + i + 6*nnml_f32_epr);
            sum7 = NNML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = NNML_F32_VEC_LOAD(x + i + 7*nnml_f32_epr);
            ay8 = NNML_F32_VEC_LOAD(y + i + 7*nnml_f32_epr);
            sum8 = NNML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, nnml_f32_step] which is handled in below loop
        const int np2 = (n & ~(nnml_f32_epr - 1));
        for (int i = np; i < np2; i += nnml_f32_epr) {
            ax1 = NNML_F32_VEC_LOAD(x + i);
            ay1 = NNML_F32_VEC_LOAD(y + i);
            sum1 = NNML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that nnml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmad_f32_m(pg, ax1, ay1, sum1);
        }
        // reduce sum1,sum2 to sum1
        NNML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #else
        const int np = (n & ~(NNML_F32_STEP - 1));

        NNML_F32_VEC sum[NNML_F32_ARR] = { NNML_F32_VEC_ZERO };

        NNML_F32_VEC ax[NNML_F32_ARR];
        NNML_F32_VEC ay[NNML_F32_ARR];

        for (int i = 0; i < np; i += NNML_F32_STEP) {
            for (int j = 0; j < NNML_F32_ARR; j++) {
                ax[j] = NNML_F32_VEC_LOAD(x + i + j*NNML_F32_EPR);
                ay[j] = NNML_F32_VEC_LOAD(y + i + j*NNML_F32_EPR);

                sum[j] = NNML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        NNML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif

    *s = sumf;
}

void nnml_vec_dot_f16(int n, float * NNML_RESTRICT s, size_t bs, nnml_fp16_t * NNML_RESTRICT x, size_t bx, nnml_fp16_t * NNML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    nnml_float sumf = 0.0;

    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = svcntb() * 8; //get vector length
        const int nnml_f16_epr = sve_register_length / 16; // running when 16
        const int nnml_f16_step = 8 * nnml_f16_epr; // choose 8 SVE registers

        const int np= (n & ~(nnml_f16_step - 1));
        svfloat16_t sum1 = svdup_n_f16(0.0f);
        svfloat16_t sum2 = svdup_n_f16(0.0f);
        svfloat16_t sum3 = svdup_n_f16(0.0f);
        svfloat16_t sum4 = svdup_n_f16(0.0f);

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += nnml_f16_step) {
            ax1 = NNML_F16x_VEC_LOAD(x + i + 0 * nnml_f16_epr, 0);
            ay1 = NNML_F16x_VEC_LOAD(y + i + 0 * nnml_f16_epr, 0);
            sum1 = NNML_F16x_VEC_FMA(sum1, ax1, ay1);

            ax2 = NNML_F16x_VEC_LOAD(x + i + 1 * nnml_f16_epr, 1);
            ay2 = NNML_F16x_VEC_LOAD(y + i + 1 * nnml_f16_epr, 1);
            sum2 = NNML_F16x_VEC_FMA(sum2, ax2, ay2);

            ax3 = NNML_F16x_VEC_LOAD(x + i + 2 * nnml_f16_epr, 2);
            ay3 = NNML_F16x_VEC_LOAD(y + i + 2 * nnml_f16_epr, 2);
            sum3 = NNML_F16x_VEC_FMA(sum3, ax3, ay3);

            ax4 = NNML_F16x_VEC_LOAD(x + i + 3 * nnml_f16_epr, 3);
            ay4 = NNML_F16x_VEC_LOAD(y + i + 3 * nnml_f16_epr, 3);
            sum4 = NNML_F16x_VEC_FMA(sum4, ax4, ay4);

            ax5 = NNML_F16x_VEC_LOAD(x + i + 4 * nnml_f16_epr, 4);
            ay5 = NNML_F16x_VEC_LOAD(y + i + 4 * nnml_f16_epr, 4);
            sum1 = NNML_F16x_VEC_FMA(sum1, ax5, ay5);

            ax6 = NNML_F16x_VEC_LOAD(x + i + 5 * nnml_f16_epr, 5);
            ay6 = NNML_F16x_VEC_LOAD(y + i + 5 * nnml_f16_epr, 5);
            sum2 = NNML_F16x_VEC_FMA(sum2, ax6, ay6);

            ax7 = NNML_F16x_VEC_LOAD(x + i + 6 * nnml_f16_epr, 6);
            ay7 = NNML_F16x_VEC_LOAD(y + i + 6 * nnml_f16_epr, 6);
            sum3 = NNML_F16x_VEC_FMA(sum3, ax7, ay7);

            ax8 = NNML_F16x_VEC_LOAD(x + i + 7 * nnml_f16_epr, 7);
            ay8 = NNML_F16x_VEC_LOAD(y + i + 7 * nnml_f16_epr, 7);
            sum4 = NNML_F16x_VEC_FMA(sum4, ax8, ay8);
        }

        const int np2 = (n & ~(nnml_f16_epr - 1)); // round down to multiple of 8
        for (int k = np; k < np2; k += nnml_f16_epr) {
            svfloat16_t rx = NNML_F16x_VEC_LOAD(x + k, 0);
            svfloat16_t ry = NNML_F16x_VEC_LOAD(y + k, 0);
            sum1 = NNML_F16x_VEC_FMA(sum1, rx, ry);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));

            sum1 = svmad_f16_x(pg, hx, hy, sum1);
        }
        NNML_F16x_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4);
    #else
        const int np = (n & ~(NNML_F16_STEP - 1));

        NNML_F16_VEC sum[NNML_F16_ARR] = { NNML_F16_VEC_ZERO };

        NNML_F16_VEC ax[NNML_F16_ARR];
        NNML_F16_VEC ay[NNML_F16_ARR];

        for (int i = 0; i < np; i += NNML_F16_STEP) {
            for (int j = 0; j < NNML_F16_ARR; j++) {
                ax[j] = NNML_F16_VEC_LOAD(x + i + j*NNML_F16_EPR, j);
                ay[j] = NNML_F16_VEC_LOAD(y + i + j*NNML_F16_EPR, j);

                sum[j] = NNML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        NNML_F16_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += (nnml_float)(NNML_CPU_FP16_TO_FP32(x[i])*NNML_CPU_FP16_TO_FP32(y[i]));
        }
        // if you hit this, you are likely running outside the FP range
        assert(!std::isnan(sumf) && !std::isinf(sumf));
    #endif

    *s = sumf;
}

#if defined(__ARM_NEON)
    #if !defined(__ARM_FEATURE_DOTPROD)
    inline static int32x4_t nnml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
        const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
        const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

        return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
    }
    #else
    #define nnml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)
    #endif
#endif

void nnml_vec_dot_q4_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    const block_q4_0 * NNML_RESTRICT x = (const block_q4_0 * NNML_RESTRICT)vx;
    const block_q8_0 * NNML_RESTRICT y = (const block_q8_0 * NNML_RESTRICT)vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * NNML_RESTRICT vx0 = vx;
        const block_q4_0 * NNML_RESTRICT vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * NNML_RESTRICT vy0 = vy;
        const block_q8_0 * NNML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * NNML_RESTRICT b_x0 = &vx0[i];
            const block_q4_0 * NNML_RESTRICT b_x1 = &vx1[i];
            const block_q8_0 * NNML_RESTRICT b_y0 = &vy0[i];
            const block_q8_0 * NNML_RESTRICT b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                NNML_CPU_FP16_TO_FP32(b_x0->d)*NNML_CPU_FP16_TO_FP32(b_y0->d),
                NNML_CPU_FP16_TO_FP32(b_x0->d)*NNML_CPU_FP16_TO_FP32(b_y1->d),
                NNML_CPU_FP16_TO_FP32(b_x1->d)*NNML_CPU_FP16_TO_FP32(b_y0->d),
                NNML_CPU_FP16_TO_FP32(b_x1->d)*NNML_CPU_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = nnml_cpu_get_sve_cnt()*8;

    // VLA Implementation using switch case
    switch (vector_length) {
        case 128:
            {
                // predicate for activating higher lanes for 4 float32 elements
                const svbool_t ph4 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx0r, 0x0F));
                    const svint8_t qx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx0r, 0x04));
                    const svint8_t qx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx1r, 0x0F));
                    const svint8_t qx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx1r, 0x04));

                    // sub 8
                    const svint8_t qx0ls = svsub_n_s8_x(svptrue_b8(), qx0h, 8);
                    const svint8_t qx0hs = svsub_n_s8_x(svptrue_b8(), qx0l, 8);
                    const svint8_t qx1ls = svsub_n_s8_x(svptrue_b8(), qx1h, 8);
                    const svint8_t qx1hs = svsub_n_s8_x(svptrue_b8(), qx1l, 8);

                    // load y
                    const svint8_t qy0h = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy0l = svld1_s8(svptrue_b8(), y0->qs + 16);
                    const svint8_t qy1h = svld1_s8(svptrue_b8(), y1->qs);
                    const svint8_t qy1l = svld1_s8(svptrue_b8(), y1->qs + 16);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph4, sumv0, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx0ls, qy0l),
                                    svdot_s32(svdup_n_s32(0), qx0hs, qy0h))), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph4, sumv1, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx1ls, qy1l),
                                    svdot_s32(svdup_n_s32(0), qx1hs, qy1h))), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 256:
            {
                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for  16 int8 elements
                const svbool_t pl16 = svnot_b_z(svptrue_b8(), ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating higher lanes for 32 int8 elements
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);

                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
                const svbool_t pl16 = svnot_b_z(ph32, ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(ph32, x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(ph32, x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(ph32, qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(ph32, qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(ph32, y0->qs);
                    const svint8_t qy1 = svld1_s8(ph32, y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph32, sumv0, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph32, sumv1, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(ph32, svadd_f32_x(ph32, sumv0, sumv1));
            } break;
        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0 * NNML_RESTRICT x0 = &x[ib + 0];
        const block_q4_0 * NNML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
        const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = nnml_vdotq_s32(nnml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = nnml_vdotq_s32(nnml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*NNML_CPU_FP16_TO_FP32(x[ib].d)*NNML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

void nnml_vec_dot_q6_K_q8_K_generic(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    const block_q6_K * NNML_RESTRICT x = (block_q6_K * NNML_RESTRICT)vx;
    const block_q8_K * NNML_RESTRICT y = (block_q8_K * NNML_RESTRICT)vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * NNML_RESTRICT q4 = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const  int8_t * NNML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * NNML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = NNML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void nnml_vec_dot_q6_K_q8_K(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
#ifdef __ARM_FEATURE_MATMUL_INT8
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    const block_q6_K * NNML_RESTRICT x = (block_q6_K * NNML_RESTRICT)vx;
    const block_q8_K * NNML_RESTRICT y = (block_q8_K * NNML_RESTRICT)vy;

    const int nb = n / QK_K;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q6_K * NNML_RESTRICT x0 = x;
        const block_q6_K * NNML_RESTRICT x1 = (const block_q6_K *) ((const uint8_t *)vx + bx);
        const block_q8_K * NNML_RESTRICT y0 = y;
        const block_q8_K * NNML_RESTRICT y1 = (const block_q8_K *) ((const uint8_t *)vy + by);

        float32x4_t vfsum = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; ++i, ++x0, ++x1, ++y0, ++y1) {
            const uint8_t * NNML_RESTRICT ql0 = x0->ql;
            const uint8_t * NNML_RESTRICT ql1 = x1->ql;
            const uint8_t * NNML_RESTRICT qh0 = x0->qh;
            const uint8_t * NNML_RESTRICT qh1 = x1->qh;
            const  int8_t * NNML_RESTRICT qy0 = y0->qs;
            const  int8_t * NNML_RESTRICT qy1 = y1->qs;

            const uint8x16_t mone = vdupq_n_u8(0x30);
            const uint8x16_t  m4b = vdupq_n_u8(0x0f);

            int32x4_t visum = vdupq_n_s32(0);

            // process 8 blocks per iteration, totally 16 blocks
            for (int j = 0; j < 2; ++j, qh0 += 32, ql0 += 64, qh1 += 32, ql1 += 64) {
                int8x16_t vx0[8], vx1[8];

                // de-quantize vx0[8]
                {
                    const uint8x16x2_t qh_bits = vld1q_u8_x2(qh0);
                    const uint8x16x4_t ql_bits = vld1q_u8_x4(ql0);

                    uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
                    uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
                    uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
                    uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

                    vx0[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
                    vx0[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
                    vx0[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
                    vx0[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

                    q6h_0 = vandq_u8(mone, qh_bits.val[0]);
                    q6h_1 = vandq_u8(mone, qh_bits.val[1]);
                    q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
                    q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

                    vx0[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
                    vx0[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
                    vx0[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
                    vx0[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
                }

                // de-quantize vx1[8]
                {
                    const uint8x16x2_t qh_bits = vld1q_u8_x2(qh1);
                    const uint8x16x4_t ql_bits = vld1q_u8_x4(ql1);

                    uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
                    uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
                    uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
                    uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

                    vx1[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
                    vx1[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
                    vx1[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
                    vx1[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

                    q6h_0 = vandq_u8(mone, qh_bits.val[0]);
                    q6h_1 = vandq_u8(mone, qh_bits.val[1]);
                    q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
                    q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

                    vx1[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
                    vx1[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
                    vx1[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
                    vx1[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
                }

                // process 16 elements (one block with same scale) per iteration
                // - vx = concat(ql, qh) - 32
                // - r1,r2,r3,r4 = smmla(vx, vy)
                for (int k = 0; k < 8; ++k) {
                    const int blk = j * 8 + k;

                    const int8x16_t vy0 = vld1q_s8(qy0);
                    const int8x16_t vy1 = vld1q_s8(qy1);
                    qy0 += 16;
                    qy1 += 16;

                    const int32x4_t block_scale = {
                        x0->scales[blk],
                        x0->scales[blk],
                        x1->scales[blk],
                        x1->scales[blk],
                    };

                    // calculate four results at once with outer product
                    const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
                    const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
                    const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
                    const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
                    int32x4_t vr = vdupq_n_s32(0);
                    vr = vmmlaq_s32(vr, vx_l, vy_l);
                    vr = vmmlaq_s32(vr, vx_h, vy_h);

                    // apply block scale, will NOT overflow
                    // block_scale * sum_256(int6*int8) <= 2^(8+8+6+8) = 30 bits
                    visum = vmlaq_s32(visum, vr, block_scale);
                }
            }

            // adjust bias, apply superblock scale
            {
                int32_t bias[4];
#ifdef __ARM_FEATURE_SVE
                const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
                const svbool_t pg8_8 = svptrue_pat_b8(SV_VL8);
                const svint16_t y0_q8sums_0 = svld1_s16(pg16_8, y0->bsums);
                const svint16_t y0_q8sums_1 = svld1_s16(pg16_8, y0->bsums + 8);
                const svint16_t y1_q8sums_0 = svld1_s16(pg16_8, y1->bsums);
                const svint16_t y1_q8sums_1 = svld1_s16(pg16_8, y1->bsums + 8);
                const svint16_t x0_q6scales_0 = svunpklo_s16(svld1_s8(pg8_8, x0->scales));
                const svint16_t x0_q6scales_1 = svunpklo_s16(svld1_s8(pg8_8, x0->scales + 8));
                const svint16_t x1_q6scales_0 = svunpklo_s16(svld1_s8(pg8_8, x1->scales));
                const svint16_t x1_q6scales_1 = svunpklo_s16(svld1_s8(pg8_8, x1->scales + 8));
                const svint64_t zero = svdup_n_s64(0);
                bias[0] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y0_q8sums_0, x0_q6scales_0),
                                                                               svdot_s64(zero, y0_q8sums_1, x0_q6scales_1)));
                bias[1] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y1_q8sums_0, x0_q6scales_0),
                                                                               svdot_s64(zero, y1_q8sums_1, x0_q6scales_1)));
                bias[2] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y0_q8sums_0, x1_q6scales_0),
                                                                               svdot_s64(zero, y0_q8sums_1, x1_q6scales_1)));
                bias[3] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y1_q8sums_0, x1_q6scales_0),
                                                                               svdot_s64(zero, y1_q8sums_1, x1_q6scales_1)));
#else
                // NEON doesn't support int16 dot product, fallback to separated mul and add
                const int16x8x2_t q8sums0 = vld1q_s16_x2(y0->bsums);
                const int16x8x2_t q8sums1 = vld1q_s16_x2(y1->bsums);

                int8x16_t scales_s8 = vld1q_s8(x0->scales);
                const int16x8x2_t q6scales0 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};
                scales_s8 = vld1q_s8(x1->scales);
                const int16x8x2_t q6scales1 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};

                int32x4_t prod;
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[0]), vget_low_s16 (q6scales0.val[0])),
                                           vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales0.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[1]), vget_low_s16 (q6scales0.val[1])),
                                           vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales0.val[1]))));
                bias[0] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[0]), vget_low_s16 (q6scales0.val[0])),
                                           vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales0.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[1]), vget_low_s16 (q6scales0.val[1])),
                                           vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales0.val[1]))));
                bias[1] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[0]), vget_low_s16 (q6scales1.val[0])),
                                           vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales1.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[1]), vget_low_s16 (q6scales1.val[1])),
                                           vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales1.val[1]))));
                bias[2] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[0]), vget_low_s16 (q6scales1.val[0])),
                                           vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales1.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[1]), vget_low_s16 (q6scales1.val[1])),
                                           vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales1.val[1]))));
                bias[3] = vaddvq_s32(prod);

#endif
                const int32x4_t vibias = vmulq_n_s32(vld1q_s32(bias), 32);

                const float32x4_t superblock_scale = {
                    NNML_CPU_FP16_TO_FP32(x0->d) * y0->d,
                    NNML_CPU_FP16_TO_FP32(x0->d) * y1->d,
                    NNML_CPU_FP16_TO_FP32(x1->d) * y0->d,
                    NNML_CPU_FP16_TO_FP32(x1->d) * y1->d,
                };

                visum = vsubq_s32(visum, vibias);
                vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
            }
        }

        // vfsum = ABCD -> ACBD
        // AC -> s, BD -> (s+bs)
        vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
        vst1_f32(s,      vget_low_f32 (vfsum));
        vst1_f32(s + bs, vget_high_f32(vfsum));

        return;
    }
#endif

#ifdef __ARM_FEATURE_SVE
    const int vector_length = nnml_cpu_get_sve_cnt()*8;
    float sum = 0;
    svuint8_t m4b = svdup_n_u8(0xf);
    svint32_t vzero = svdup_n_s32(0);
    svuint8_t mone = svdup_n_u8(0x30);
    svint8_t q6bytes_1, q6bytes_2, q6bytes_3, q6bytes_4;
    svuint8_t q6h_1, q6h_2, q6h_3, q6h_4;

    for (int i = 0; i < nb; ++i) {
        const float d_all = NNML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * NNML_RESTRICT q6 = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const int8_t  * NNML_RESTRICT q8 = y[i].qs;

        const int8_t * NNML_RESTRICT scale = x[i].scales;

        const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
        const svint16_t q8sums_1 = svld1_s16(pg16_8, y[i].bsums);
        const svint16_t q8sums_2 = svld1_s16(pg16_8, y[i].bsums + 8);
        const svint16_t q6scales_1 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale));
        const svint16_t q6scales_2 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale + 8));
        const svint64_t prod = svdup_n_s64(0);
        int32_t isum_mins = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(prod, q8sums_1, q6scales_1),
                                                                                 svdot_s64(prod, q8sums_2, q6scales_2)));
        int32_t isum = 0;

        switch (vector_length) {
            case 128:
                {
                    const svbool_t pg32_4 = svptrue_pat_b32(SV_VL4);
                    const svbool_t pg8_16 = svptrue_pat_b8(SV_VL16);
                    svint32_t isum_tmp = svdup_n_s32(0);
                    for (int j = 0; j < QK_K/128; ++j) {
                        svuint8_t qhbits_1 = svld1_u8(pg8_16, qh);
                        svuint8_t qhbits_2 = svld1_u8(pg8_16, qh+16);
                        qh += 32;
                        svuint8_t q6bits_1 = svld1_u8(pg8_16, q6);
                        svuint8_t q6bits_2 = svld1_u8(pg8_16, q6+16);
                        svuint8_t q6bits_3 = svld1_u8(pg8_16, q6+32);
                        svuint8_t q6bits_4 = svld1_u8(pg8_16, q6+48);
                        q6 += 64;
                        svint8_t q8bytes_1 = svld1_s8(pg8_16, q8);
                        svint8_t q8bytes_2 = svld1_s8(pg8_16, q8+16);
                        svint8_t q8bytes_3 = svld1_s8(pg8_16, q8+32);
                        svint8_t q8bytes_4 = svld1_s8(pg8_16, q8+48);
                        q8 += 64;

                        q6h_1 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 4));
                        q6h_2 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 4));
                        q6h_3 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 2));
                        q6h_4 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_1, m4b), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_2, m4b), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_3, m4b), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_4, m4b), q6h_4));
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);

                        scale += 4;
                        q8bytes_1 = svld1_s8(pg8_16, q8);
                        q8bytes_2 = svld1_s8(pg8_16, q8+16);
                        q8bytes_3 = svld1_s8(pg8_16, q8+32);
                        q8bytes_4 = svld1_s8(pg8_16, q8+48);
                        q8 += 64;

                        q6h_1 = svand_u8_x(pg16_8, mone, qhbits_1);
                        q6h_2 = svand_u8_x(pg16_8, mone, qhbits_2);
                        q6h_3 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_1, 2));
                        q6h_4 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_2, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_1, 4), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_2, 4), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_3, 4), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_4, 4), q6h_4));
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);
                        scale += 4;
                    }
                    isum += svaddv_s32(pg32_4, isum_tmp);
                    sum += d_all * y[i].d * (isum - 32 * isum_mins);
                }
                break;
            case 256:
            case 512:
                {
                    const svbool_t pg8_2 = svptrue_pat_b8(SV_VL2);
                    const svbool_t pg32_8 = svptrue_pat_b32(SV_VL8);
                    const svbool_t pg8_32 = svptrue_pat_b8(SV_VL32);
                    svint32_t isum_tmp = svdup_n_s32(0);
                    for (int j = 0; j < QK_K/128; j++) {
                        svuint8_t qhbits_1 = svld1_u8(pg8_32, qh);
                        qh += 32;
                        svuint8_t q6bits_1 = svld1_u8(pg8_32, q6);
                        svuint8_t q6bits_2 = svld1_u8(pg8_32, q6+32);
                        q6 += 64;
                        svint8_t q8bytes_1 = svld1_s8(pg8_32, q8);
                        svint8_t q8bytes_2 = svld1_s8(pg8_32, q8+32);
                        svint8_t q8bytes_3 = svld1_s8(pg8_32, q8+64);
                        svint8_t q8bytes_4 = svld1_s8(pg8_32, q8+96);
                        q8 += 128;
                        q6h_1 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 4));
                        q6h_2 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 2));
                        q6h_3 = svand_u8_x(pg8_32, mone, qhbits_1);
                        q6h_4 = svand_u8_x(pg8_32, mone, svlsr_n_u8_x(pg8_32, qhbits_1, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_1, m4b), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_2, m4b), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_1, 4), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_2, 4), q6h_4));

                        svint8_t scale_lane_1_tmp = svld1_s8(pg8_2, scale);
                        scale_lane_1_tmp= svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
                        scale_lane_1_tmp= svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
                        svint8_t scale_lane_2_tmp = svld1_s8(pg8_2, scale+2);
                        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
                        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
                        svint8_t scale_lane_3_tmp = svld1_s8(pg8_2, scale+4);
                        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
                        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
                        svint8_t scale_lane_4_tmp = svld1_s8(pg8_2, scale+6);
                        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
                        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
                        svint32_t scale_lane_1 = svunpklo_s32(svunpklo_s16(scale_lane_1_tmp));
                        svint32_t scale_lane_2 = svunpklo_s32(svunpklo_s16(scale_lane_2_tmp));
                        svint32_t scale_lane_3 = svunpklo_s32(svunpklo_s16(scale_lane_3_tmp));
                        svint32_t scale_lane_4 = svunpklo_s32(svunpklo_s16(scale_lane_4_tmp));

                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale_lane_1);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale_lane_2);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale_lane_3);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale_lane_4);
                        scale += 8;
                    }
                    isum += svaddv_s32(pg32_8, isum_tmp);
                    sum += d_all * y[i].d * (isum - 32 * isum_mins);
                }
                break;
            default:
                assert(false && "Unsupported vector length");
                break;
        }
    }

    *s = sum;

#elif __ARM_NEON
    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    //const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    nnml_int8x16x4_t q6bytes;
    nnml_uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = NNML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * NNML_RESTRICT q6 = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const int8_t  * NNML_RESTRICT q8 = y[i].qs;

        const int8_t * NNML_RESTRICT scale = x[i].scales;

        const nnml_int16x8x2_t q8sums = nnml_vld1q_s16_x2(y[i].bsums);
        const int8x16_t scales = vld1q_s8(scale);
        const nnml_int16x8x2_t q6scales = {{vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

        const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[0]), vget_low_s16 (q6scales.val[0])),
                                                   vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                         vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[1]), vget_low_s16 (q6scales.val[1])),
                                                   vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        int32_t isum_mins = vaddvq_s32(prod);

        int32_t isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            nnml_uint8x16x2_t qhbits = nnml_vld1q_u8_x2(qh); qh += 32;
            nnml_uint8x16x4_t q6bits = nnml_vld1q_u8_x4(q6); q6 += 64;
            nnml_int8x16x4_t q8bytes = nnml_vld1q_s8_x4(q8); q8 += 64;
            
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

            isum += vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];

            scale += 4;

            q8bytes = nnml_vld1q_s8_x4(q8); q8 += 64;

            shifted = vshrq_n_u8(qhbits.val[0], 4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[0], 6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

            isum += vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(nnml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;
        }
        //sum += isum * d_all * y[i].d;
        sum += d_all * y[i].d * (isum - 32 * isum_mins);

    }
    *s = sum;
#else
    NNML_UNUSED(x);
    NNML_UNUSED(y);
    NNML_UNUSED(nb);
    nnml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void nnml_vec_dot_q8_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);
    const block_q8_0 * NNML_RESTRICT x = (block_q8_0 * NNML_RESTRICT)vx;
    const block_q8_0 * NNML_RESTRICT y = (block_q8_0 * NNML_RESTRICT)vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0 * NNML_RESTRICT vx0 = (block_q8_0 * NNML_RESTRICT)vx;
        const block_q8_0 * NNML_RESTRICT vx1 = (const block_q8_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * NNML_RESTRICT vy0 = (block_q8_0 * NNML_RESTRICT)vy;
        const block_q8_0 * NNML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 * NNML_RESTRICT b_x0 = &vx0[i];
            const block_q8_0 * NNML_RESTRICT b_y0 = &vy0[i];

            const block_q8_0 * NNML_RESTRICT b_x1 = &vx1[i];
            const block_q8_0 * NNML_RESTRICT b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                NNML_CPU_FP16_TO_FP32(b_x0->d)*NNML_CPU_FP16_TO_FP32(b_y0->d),
                NNML_CPU_FP16_TO_FP32(b_x0->d)*NNML_CPU_FP16_TO_FP32(b_y1->d),
                NNML_CPU_FP16_TO_FP32(b_x1->d)*NNML_CPU_FP16_TO_FP32(b_y0->d),
                NNML_CPU_FP16_TO_FP32(b_x1->d)*NNML_CPU_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = nnml_cpu_get_sve_cnt()*8;

    //VLA Implemenation for SVE
    switch (vector_length) {
        case 128:
            {
                // predicate for activating lanes for 16 Int8 elements
                const svbool_t ph16 = svptrue_pat_b8 (SV_VL16);
                const svbool_t pl16 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
                    const svint8_t qx0_1 = svld1_s8(ph16, x0->qs+16);
                    const svint8_t qx1_0 = svld1_s8(ph16, x1->qs);
                    const svint8_t qx1_1 = svld1_s8(ph16, x1->qs+16);

                    // load y
                    const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
                    const svint8_t qy0_1 = svld1_s8(ph16, y0->qs+16);
                    const svint8_t qy1_0 = svld1_s8(ph16, y1->qs);
                    const svint8_t qy1_1 = svld1_s8(ph16, y1->qs+16);

                    sumv0 = svmla_n_f32_x(pl16, sumv0, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
                                    svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(pl16, sumv1, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx1_0, qy1_0),
                                    svdot_s32(svdup_n_s32(0), qx1_1, qy1_1))), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));
            } break;
        case 256:
            {
                //printf("sve256");
                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
                    const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0, qy0)), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1, qy1)), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating high 256 bit
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);
                // predicate for activating low 256 bit
                const svbool_t pl32 = svnot_b_z(svptrue_b8(), ph32);

                // predicate for activating high lanes for 8 float32 elements
                const svbool_t ph8 = svptrue_pat_b32(SV_VL8);
                // predicate for activating low lanes for 8 float32 elements
                const svbool_t pl8 = svnot_b_z(svptrue_b32(), ph8);

                svfloat32_t sumv00 = svdup_n_f32(0.0f);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * NNML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * NNML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

                    //load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
                    // and add them to make one 64 element vector
                    // load x
                    const svint8_t qx_32 = svld1_s8(ph32, x0->qs);
                          svint8_t qx_64 = svld1_s8(pl32, x0->qs + 2);

                    qx_64 = svadd_s8_x(svptrue_b8(), qx_32, qx_64);

                    // load y
                    const svint8_t qy_32 = svld1_s8(ph32, y0->qs);
                          svint8_t qy_64 = svld1_s8(pl32, y0->qs + 2);

                    qy_64 = svadd_s8_x(svptrue_b8(), qy_32, qy_64);

                    // scale creation
                    const float32_t deq1 = NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d);
                    const float32_t deq2 = NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d);

                    // duplicate deq1 in first half of vector and deq2 in second half of vector
                    const svfloat32_t temp = svdup_f32_m(svdup_f32_z(ph8, deq1), pl8, deq2);

                    const svfloat32_t sumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx_64, qy_64));

                    sumv00 = svmla_f32_m(svptrue_b32(), sumv00, sumvt, temp);
                }

                sumf = svaddv_f32(svptrue_b32(), sumv00);
                break;
            }
        default:
            assert(false && "Unsupported vector length");
            break;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 * NNML_RESTRICT x0 = &x[ib + 0];
        const block_q8_0 * NNML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * NNML_RESTRICT y0 = &y[ib + 0];
        const block_q8_0 * NNML_RESTRICT y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        nnml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        nnml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), NNML_CPU_FP16_TO_FP32(x0->d)*NNML_CPU_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        nnml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        nnml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), NNML_CPU_FP16_TO_FP32(x1->d)*NNML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(NNML_CPU_FP16_TO_FP32(x[ib].d)*NNML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void nnml_vec_scale_f32(const int n, float * y, const float v) {
#if defined(__ARM_FEATURE_SVE)
    const int sve_register_length = nnml_cpu_get_sve_cnt() * 8;
    const int nnml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
    const int nnml_f32_step = 2 * nnml_f32_epr;

    NNML_F32_VEC vx = NNML_F32_VEC_SET1(v);
    const int np = (n & ~(nnml_f32_step - 1));
    svfloat32_t ay1;
    svfloat32_t ay2;
    for (int i = 0; i < np; i += nnml_f32_step) {
        ay1 = NNML_F32_VEC_LOAD(y + i);
        ay1 = NNML_F32_VEC_MUL(ay1, vx);
        NNML_F32_VEC_STORE(y + i, ay1);

        ay2 = NNML_F32_VEC_LOAD(y + i + 1*nnml_f32_epr);
        ay2 = NNML_F32_VEC_MUL(ay2, vx);
        NNML_F32_VEC_STORE(y + i + 1*nnml_f32_epr, ay2);
    }
    // leftovers
    // maximum number of leftover elements will be less that nnml_f32_epr. Apply predicated svmad on available elements only
    if (np < n) {
        svbool_t pg = svwhilelt_b32(np, n);
        ay1 = svld1_f32(pg, y + np);
        ay1 = svmul_f32_m(pg, ay1, vx);
        svst1_f32(pg, y + np, ay1);
    }
#else
    const int np = (n & ~(NNML_F32_STEP - 1));

    NNML_F32_VEC vx = NNML_F32_VEC_SET1(v);

    NNML_F32_VEC ay[NNML_F32_ARR];

    for (int i = 0; i < np; i += NNML_F32_STEP) {
        for (int j = 0; j < NNML_F32_ARR; j++) {
            ay[j] = NNML_F32_VEC_LOAD(y + i + j*NNML_F32_EPR);
            ay[j] = NNML_F32_VEC_MUL(ay[j], vx);

            NNML_F32_VEC_STORE(y + i + j*NNML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void nnml_vec_scale_f16(const int n, nnml_fp16_t * y, const float v) {
#if defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int nnml_f16_epr = sve_register_length / 16;
    const int nnml_f16_step = 2 * nnml_f16_epr;

    NNML_F16x_VEC vx =  NNML_F16x_VEC_SET1(v);
    const int np = (n & ~(nnml_f16_step - 1));
    svfloat16_t ay1, ay2;

    for (int i = 0; i < np; i += nnml_f16_step) {
        ay1 = NNML_F16x_VEC_LOAD(y + i + 0*nnml_f16_epr, 0);
        ay1 = NNML_F16x_VEC_MUL(ay1, vx);
        NNML_F16x_VEC_STORE(y + i + 0*nnml_f16_epr, ay1, 0);

        ay2 = NNML_F16x_VEC_LOAD(y + i + 1*nnml_f16_epr, 1);
        ay2 = NNML_F16x_VEC_MUL(ay2, vx);
        NNML_F16x_VEC_STORE(y + i + 1*nnml_f16_epr, ay2, 1);
    }
    // leftovers
    // maximum number of leftover elements will be less that nnmlF_16x_epr. Apply predicated svmad on available elements only
    if (np < n) {
        svbool_t pg = svwhilelt_b16(np, n);
        svfloat16_t hy = svld1_f16(pg, (__fp16 *)(y + np));
        svfloat16_t out = svmul_f16_m(pg, hy, vx);
        svst1_f16(pg, (__fp16 *)(y + np), out);
    }
#else
    const int np = (n & ~(NNML_F16_STEP - 1));

    NNML_F16_VEC vx = NNML_F16_VEC_SET1(v);

    NNML_F16_VEC ay[NNML_F16_ARR];

    for (int i = 0; i < np; i += NNML_F16_STEP) {
        for (int j = 0; j < NNML_F16_ARR; j++) {
            ay[j] = NNML_F16_VEC_LOAD(y + i + j*NNML_F16_EPR, j);
            ay[j] = NNML_F16_VEC_MUL(ay[j], vx);

            NNML_F16_VEC_STORE(y + i + j*NNML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = NNML_CPU_FP32_TO_FP16(NNML_CPU_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void nnml_vec_mad_f32(const int n, float * NNML_RESTRICT y, const float * NNML_RESTRICT x, const float v) {
#if defined(__ARM_FEATURE_SVE)
    const int sve_register_length = nnml_cpu_get_sve_cnt() * 8;
    const int nnml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
    const int nnml_f32_step = 8 * nnml_f32_epr; // choose 8 SVE registers
    NNML_F32_VEC vx = NNML_F32_VEC_SET1(v);

    const int np = (n & ~(nnml_f32_step - 1));
    svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
    svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
    for (int i = 0; i < np; i += nnml_f32_step) {

        ax1 = NNML_F32_VEC_LOAD(x + i);
        ay1 = NNML_F32_VEC_LOAD(y + i);
        ay1 = NNML_F32_VEC_FMA(ay1, ax1, vx);

        NNML_F32_VEC_STORE(y + i, ay1);

        ax2 = NNML_F32_VEC_LOAD(x + i + 1*nnml_f32_epr);
        ay2 = NNML_F32_VEC_LOAD(y + i + 1*nnml_f32_epr);
        ay2 = NNML_F32_VEC_FMA(ay2, ax2, vx);

        NNML_F32_VEC_STORE(y + i + 1*nnml_f32_epr, ay2);

        ax3 = NNML_F32_VEC_LOAD(x + i + 2*nnml_f32_epr);
        ay3 = NNML_F32_VEC_LOAD(y + i + 2*nnml_f32_epr);
        ay3 = NNML_F32_VEC_FMA(ay3, ax3, vx);

        NNML_F32_VEC_STORE(y + i + 2*nnml_f32_epr, ay3);

        ax4 = NNML_F32_VEC_LOAD(x + i + 3*nnml_f32_epr);
        ay4 = NNML_F32_VEC_LOAD(y + i + 3*nnml_f32_epr);
        ay4 = NNML_F32_VEC_FMA(ay4, ax4, vx);

        NNML_F32_VEC_STORE(y + i + 3*nnml_f32_epr, ay4);

        ax5 = NNML_F32_VEC_LOAD(x + i + 4*nnml_f32_epr);
        ay5 = NNML_F32_VEC_LOAD(y + i + 4*nnml_f32_epr);
        ay5 = NNML_F32_VEC_FMA(ay5, ax5, vx);

        NNML_F32_VEC_STORE(y + i + 4*nnml_f32_epr, ay5);

        ax6 = NNML_F32_VEC_LOAD(x + i + 5*nnml_f32_epr);
        ay6 = NNML_F32_VEC_LOAD(y + i + 5*nnml_f32_epr);
        ay6 = NNML_F32_VEC_FMA(ay6, ax6, vx);

        NNML_F32_VEC_STORE(y + i + 5*nnml_f32_epr, ay6);

        ax7 = NNML_F32_VEC_LOAD(x + i + 6*nnml_f32_epr);
        ay7 = NNML_F32_VEC_LOAD(y + i + 6*nnml_f32_epr);
        ay7 = NNML_F32_VEC_FMA(ay7, ax7, vx);

        NNML_F32_VEC_STORE(y + i + 6*nnml_f32_epr, ay7);

        ax8 = NNML_F32_VEC_LOAD(x + i + 7*nnml_f32_epr);
        ay8 = NNML_F32_VEC_LOAD(y + i + 7*nnml_f32_epr);
        ay8 = NNML_F32_VEC_FMA(ay8, ax8, vx);

        NNML_F32_VEC_STORE(y + i + 7*nnml_f32_epr, ay8);
    }
    // leftovers
    // Since 8 unrolls are done in above loop, leftovers lie in range [0, nnml_f32_step] which is handled in below loop
    const int np2 = (n & ~(nnml_f32_epr - 1));
    for (int i = np; i < np2; i += nnml_f32_epr) {
        ax1 = NNML_F32_VEC_LOAD(x + i);
        ay1 = NNML_F32_VEC_LOAD(y + i);
        ay1 = NNML_F32_VEC_FMA(ay1, ax1, vx);

        NNML_F32_VEC_STORE(y + i, ay1);
    }
    // maximum number of leftover elements will be less that nnml_f32_epr. Apply predicated svmad on available elements only
    if (np2 < n) {
        svbool_t pg =svwhilelt_b32(np2, n);
        ax1 = svld1_f32(pg, x + np2);
        ay1 = svld1_f32(pg, y + np2);
        ay1 = svmad_f32_m(pg, ax1, vx, ay1);

        svst1_f32(pg, y + np2, ay1);
    }
#else
    const int np = (n & ~(NNML_F32_STEP - 1));
    NNML_F32_VEC vx = NNML_F32_VEC_SET1(v);
    NNML_F32_VEC ax[NNML_F32_ARR];
    NNML_F32_VEC ay[NNML_F32_ARR];

    for (int i = 0; i < np; i += NNML_F32_STEP) {
        for (int j = 0; j < NNML_F32_ARR; j++) {
            ax[j] = NNML_F32_VEC_LOAD(x + i + j*NNML_F32_EPR);
            ay[j] = NNML_F32_VEC_LOAD(y + i + j*NNML_F32_EPR);
            ay[j] = NNML_F32_VEC_FMA(ay[j], ax[j], vx);

            NNML_F32_VEC_STORE(y + i + j*NNML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void nnml_vec_mad_f16(const int n, nnml_fp16_t * NNML_RESTRICT y, const nnml_fp16_t * NNML_RESTRICT x, const float v) {
#if defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int nnml_f16_epr = sve_register_length / 16;
    const int nnml_f16_step = 8 * nnml_f16_epr;

    NNML_F16x_VEC vx = NNML_F16x_VEC_SET1(v);

    const int np= (n & ~(nnml_f16_step - 1));

    svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
    svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
    for (int i = 0; i < np; i += nnml_f16_step) {
        ax1 = NNML_F16x_VEC_LOAD(x + i + 0 * nnml_f16_epr, 0);
        ay1 = NNML_F16x_VEC_LOAD(y + i + 0 * nnml_f16_epr, 0);
        ay1 = NNML_F16x_VEC_FMA(ay1, ax1, vx);

        NNML_F16x_VEC_STORE(y + i + 0 * nnml_f16_epr, ay1, 0);

        ax2 = NNML_F16x_VEC_LOAD(x + i + 1 * nnml_f16_epr, 1);
        ay2 = NNML_F16x_VEC_LOAD(y + i + 1 * nnml_f16_epr, 1);
        ay2 = NNML_F16x_VEC_FMA(ay2, ax2, vx);

        NNML_F16x_VEC_STORE(y + i + 1 * nnml_f16_epr, ay2, 1);

        ax3 = NNML_F16x_VEC_LOAD(x + i + 2 * nnml_f16_epr, 2);
        ay3 = NNML_F16x_VEC_LOAD(y + i + 2 * nnml_f16_epr, 2);
        ay3 = NNML_F16x_VEC_FMA(ay3, ax3, vx);

        NNML_F16x_VEC_STORE(y + i + 2 * nnml_f16_epr, ay3, 2);

        ax4 = NNML_F16x_VEC_LOAD(x + i + 3 * nnml_f16_epr, 3);
        ay4 = NNML_F16x_VEC_LOAD(y + i + 3 * nnml_f16_epr, 3);
        ay4 = NNML_F16x_VEC_FMA(ay4, ax4, vx);

        NNML_F16x_VEC_STORE(y + i + 3 * nnml_f16_epr, ay4, 3);

        ax5 = NNML_F16x_VEC_LOAD(x + i + 4 * nnml_f16_epr, 4);
        ay5 = NNML_F16x_VEC_LOAD(y + i + 4 * nnml_f16_epr, 4);
        ay5 = NNML_F16x_VEC_FMA(ay5, ax5, vx);

        NNML_F16x_VEC_STORE(y + i + 4 * nnml_f16_epr, ay5, 4);

        ax6 = NNML_F16x_VEC_LOAD(x + i + 5 * nnml_f16_epr, 5);
        ay6 = NNML_F16x_VEC_LOAD(y + i + 5 * nnml_f16_epr, 5);
        ay6 = NNML_F16x_VEC_FMA(ay6, ax6, vx);

        NNML_F16x_VEC_STORE(y + i + 5 * nnml_f16_epr, ay6, 5);

        ax7 = NNML_F16x_VEC_LOAD(x + i + 6 * nnml_f16_epr, 6);
        ay7 = NNML_F16x_VEC_LOAD(y + i + 6 * nnml_f16_epr, 6);
        ay7 = NNML_F16x_VEC_FMA(ay7, ax7, vx);

        NNML_F16x_VEC_STORE(y + i + 6 * nnml_f16_epr, ay7, 6);

        ax8 = NNML_F16x_VEC_LOAD(x + i + 7 * nnml_f16_epr, 7);
        ay8 = NNML_F16x_VEC_LOAD(y + i + 7 * nnml_f16_epr, 7);
        ay8 = NNML_F16x_VEC_FMA(ay8, ax8, vx);

        NNML_F16x_VEC_STORE(y + i + 7 * nnml_f16_epr, ay8, 7);
    }
    const int np2 = (n & ~(nnml_f16_epr - 1));
    for (int k = np; k < np2; k += nnml_f16_epr) {
        svfloat16_t rx = NNML_F16x_VEC_LOAD(x + k, 0);
        svfloat16_t ry = NNML_F16x_VEC_LOAD(y + k, 0);
        ry = NNML_F16x_VEC_FMA(ry, rx, vx);

        NNML_F16x_VEC_STORE(y + k, ry, 0);
    }

    if (np2 < n) {
        svbool_t pg = svwhilelt_b16(np2, n);
        svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
        svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));
        hy = svmad_f16_x(pg, hx, vx, hy);
        svst1_f16(pg, (__fp16 *)(y + np2), hy);
    }
#else
    const int np = (n & ~(NNML_F16_STEP - 1));

    NNML_F16_VEC vx = NNML_F16_VEC_SET1(v);

    NNML_F16_VEC ax[NNML_F16_ARR];
    NNML_F16_VEC ay[NNML_F16_ARR];

    for (int i = 0; i < np; i += NNML_F16_STEP) {
        for (int j = 0; j < NNML_F16_ARR; j++) {
            ax[j] = NNML_F16_VEC_LOAD(x + i + j*NNML_F16_EPR, j);
            ay[j] = NNML_F16_VEC_LOAD(y + i + j*NNML_F16_EPR, j);
            ay[j] = NNML_F16_VEC_FMA(ay[j], ax[j], vx);

            NNML_F16_VEC_STORE(y + i + j*NNML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = NNML_CPU_FP32_TO_FP16(NNML_CPU_FP16_TO_FP32(y[i]) + NNML_CPU_FP16_TO_FP32(x[i])*v);
    }
#endif
}

inline static float32x4_t nnml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

inline static float32x4_t nnml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = nnml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

inline static float nnml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

void nnml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, nnml_v_silu(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(nnml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = nnml_silu_f32(x[i]) * g[i];
    }
}

void nnml_vec_swiglu_f16(const int n, nnml_fp16_t * y, const nnml_fp16_t * x, const nnml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = NNML_CPU_FP16_TO_FP32(x[i]);
        float gi = NNML_CPU_FP16_TO_FP32(g[i]);
        y[i] = NNML_CPU_FP32_TO_FP16((xi/(1.0f + expf(-xi))) * gi);
    }
}

void quantize_row_q8_0(const float * NNML_RESTRICT x, void * NNML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;
    block_q8_0 * NNML_RESTRICT y = (block_q8_0 * NNML_RESTRICT)vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];
        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);
        const float amax = vmaxvq_f32(amaxv[0]);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = NNML_CPU_FP32_TO_FP16(d);
        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);
            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    NNML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, nnml_type PARAM_TYPE>
void forward_mul_mat(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    NNML_ASSERT(ne0 == ne01);
    NNML_ASSERT(ne1 == ne11);
    NNML_ASSERT(ne2 == ne12);
    NNML_ASSERT(ne3 == ne13);

    // dst cannot be transposed or permuted
    NNML_ASSERT(nb0 == sizeof(float));
    NNML_ASSERT(nb0 <= nb1);
    NNML_ASSERT(nb1 <= nb2);
    NNML_ASSERT(nb2 <= nb3);

    NNML_ASSERT(src1->get_data_type() == NNML_TYPE_F32);

    NNML_ASSERT(src0->n_dims() == 2);
    // NNML_ASSERT(nnml_n_dims(op->src[1]) == 2);

    char *       wdata = (char *)params->work_data;
    const size_t nbw1  = nnml_row_size(PARAM_TYPE, ne10);

    assert(params->work_size >= nbw1 * ne11);

    // printf("param type: %d\n", PARAM_TYPE);
    const nnml_from_float_t from_float = nnml_get_type_traits(PARAM_TYPE)->from_float_cpu;

    int64_t i11_processed = 0;
    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
        nnml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((float *) ((char *) src1->tensor_data() + i11 * nb11), (void *) (wdata + i11 * nbw1), 4, ne10);
    }

    i11_processed = ne11 - ne11 % 4;
    for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
        from_float((float *) ((char *) src1->tensor_data() + i11 * nb11), (void *) (wdata + i11 * nbw1), ne10);
    }

    nnml_barrier_global(params->threadgrp->parent_pool);

    const void * src1_wdata      = params->work_data;
    const size_t src1_col_stride = nnml_row_size(PARAM_TYPE, ne10);
    int64_t      src0_start      = (ith * ne01) / nth;
    int64_t      src0_end        = ((ith + 1) * ne01) / nth;
    src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
    src0_end   = (src0_end   % NB_COLS) ? src0_end   + NB_COLS - (src0_end   % NB_COLS) : src0_end;
    if (src0_start >= src0_end) {
        return;
    }

    // If there are more than three rows in src1, use gemm; otherwise, use gemv.
    if (ne11 > 3) {
        gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00,
                (float *) ((char *) node->tensor_data()) + src0_start, ne01,
                (const char *) src0->tensor_data() + src0_start * nb01,
                (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
    }
    for (int iter = ne11 - ne11 % 4; iter < ne11; iter++) {
        gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00,
                (float *) ((char *) node->tensor_data() + (iter * nb1)) + src0_start, ne01,
                (const char *) src0->tensor_data() + src0_start * nb01,
                (const char *) src1_wdata + (src1_col_stride * iter), 1,
                src0_end - src0_start);
    }
}
template void forward_mul_mat<block_q4_0, 4, 4, NNML_TYPE_Q8_0>(nnml_tensor*, const nnml_compute_state*);
template void forward_mul_mat<block_q4_0, 8, 4, NNML_TYPE_Q8_0>(nnml_tensor*, const nnml_compute_state*);
template void forward_mul_mat<block_q4_0, 8, 8, NNML_TYPE_Q8_0>(nnml_tensor*, const nnml_compute_state*);

// softmax

static inline nnml_float nnml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    nnml_float sum = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svfloat32_t val = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, x + i),
                                                svdup_n_f32_x(pg, max)));
        svst1_f32(pg, y + i, val);
        sum += (ggml_float)svaddv_f32(pg, val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = nnml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (nnml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (nnml_float)val;
        y[i] = val;
    }
    return sum;
}

void nnml_compute_forward_soft_max_f32(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    const nnml_tensor * src2 = node->get_src_tensor(2);
    assert(node->is_contiguous());
    assert(are_same_shape(src0, node));
    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) node->get_operation_params() + 0, sizeof(float));
    memcpy(&max_bias, (float *) node->get_operation_params() + 1, sizeof(float));
    const int ith = params->ith;
    const int nth = params->nth;
    NNML_TENSOR_UNARY_OP_LOCALS
    const int64_t nb11 = src1 ? src1->get_stride_bytes(1) : 1;
    const int64_t nb12 = src1 ? src1->get_stride_bytes(2) : 1;
    const int64_t nb13 = src1 ? src1->get_stride_bytes(3) : 1;
    const int64_t ne12 = src1 ? src1->get_elements(2) : 1;
    const int64_t ne13 = src1 ? src1->get_elements(3) : 1;

    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    float * wp = (float *) params->work_data + (ne00 + CACHE_LINE_SIZE_F32) * ith;
    const bool use_f16 = (src1 && src1->get_data_type() == NNML_TYPE_F16);

    // sinks
    const float * sk = src2 ? (float *)((char *) src2->tensor_data()) : nullptr;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const int64_t i11 = i01;
                const int64_t i12 = i02%ne12;
                const int64_t i13 = i03%ne13;

                // ALiBi
                const uint32_t h = i02; // head
                const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                float * sp = (float *)((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03);
                float * dp = (float *)((char *) node->tensor_data() + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                nnml_fp16_t * mp_f16 = src1 ? (nnml_fp16_t *)((char *) src1->tensor_data() + i11*nb11 + i12*nb12 + i13*nb13) : NULL;
                float       * mp_f32 = src1 ? (float       *)((char *) src1->tensor_data() + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

                nnml_vec_cpy_f32  (ne00, wp, sp);
                nnml_vec_scale_f32(ne00, wp, scale);
                if (mp_f32) {
                    if (use_f16) {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope*NNML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } else {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                for (int i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    assert(!std::isnan(wp[i]));
                }
#endif

                float max = -INFINITY;
                nnml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                if (sk) {
                    max = MAX(max, sk[i02]);
                }

                nnml_float sum = nnml_vec_soft_max_f32(ne00, dp, wp, max);
                assert(sum > 0.0);

                if (sk) {
                    sum += (nnml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                nnml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                for (int i = 0; i < ne00; ++i) {
                    assert(!std::isnan(dp[i]));
                    assert(!std::isinf(dp[i]));
                }
#endif
            }
        }
    }
}

// rope

void nnml_compute_forward_rope_f16(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    const nnml_tensor * src2 = node->get_src_tensor(1);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];

    //const int n_past     = ((int32_t *) node->get_operation_params())[0];
    const int n_dims     = ((int32_t *) node->get_operation_params())[1];
    const int mode       = ((int32_t *) node->get_operation_params())[2];
    //const int n_ctx      = ((int32_t *) node->get_operation_params())[3];
    const int n_ctx_orig = ((int32_t *) node->get_operation_params())[4];
    memcpy(&freq_base,   (int32_t *) node->get_operation_params() +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) node->get_operation_params() +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) node->get_operation_params() +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) node->get_operation_params() +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) node->get_operation_params() +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) node->get_operation_params() + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) node->get_operation_params() + 11, sizeof(int)*4);


    NNML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    NNML_ASSERT(nb0 == sizeof(nnml_fp16_t));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = node->n_rows();

    NNML_ASSERT(n_dims <= ne0);
    NNML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    nnml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & NNML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & NNML_ROPE_TYPE_MROPE;

    if (is_mrope) {
        NNML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        NNML_ASSERT(src2->get_data_type() == NNML_TYPE_F32);
        NNML_ASSERT(src2->get_elements(0) >= n_dims / 2);
        freq_factors = (const float *) src2->tensor_data();
    }

    const float sin_sign = 1.0f;

    const int32_t * pos = (const int32_t *) src1->tensor_data();

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {

            float * cache = (float *) params->work_data + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            if (!is_mrope) {
                const int64_t p = pos[i2];
                nnml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                nnml_mrope_cache_init(
                    p_t, p_h, p_w, p_e, sections, false,
                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (is_neox || is_mrope) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const nnml_fp16_t * const src = (nnml_fp16_t *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        nnml_fp16_t * dst_data  = (nnml_fp16_t *)((char *)  node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = NNML_CPU_FP16_TO_FP32(src[0]);
                        const float x1 = NNML_CPU_FP16_TO_FP32(src[n_dims/2]);

                        dst_data[0]        = NNML_CPU_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims/2] = NNML_CPU_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const nnml_fp16_t * const src = (nnml_fp16_t *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              nnml_fp16_t * dst_data  = (nnml_fp16_t *)((char *) node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = NNML_CPU_FP16_TO_FP32(src[0]);
                        const float x1 = NNML_CPU_FP16_TO_FP32(src[1]);

                        dst_data[0] = NNML_CPU_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = NNML_CPU_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                }
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const nnml_fp16_t * const src = (nnml_fp16_t *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    nnml_fp16_t * dst_data  = (nnml_fp16_t *)((char *)  node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

void nnml_compute_forward_flash_attn_ext_f16(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * q     = node->get_src_tensor(0);
    const nnml_tensor * k     = node->get_src_tensor(1);
    // printf("k type ext: %d\n", k->get_data_type());
    const nnml_tensor * v     = node->get_src_tensor(2);
    const nnml_tensor * mask  = node->get_src_tensor(3);
    const nnml_tensor * sinks = node->get_src_tensor(4);

    NNML_TENSOR_LOCALS(int64_t, neq, q,   get_elements)
    NNML_TENSOR_LOCALS(size_t,  nbq, q,   get_stride_bytes)
    NNML_TENSOR_LOCALS(int64_t, nek, k,   get_elements)
    NNML_TENSOR_LOCALS(size_t,  nbk, k,   get_stride_bytes)
    NNML_TENSOR_LOCALS(int64_t, nev, v,   get_elements)
    NNML_TENSOR_LOCALS(size_t,  nbv, v,   get_stride_bytes)
    NNML_TENSOR_LOCALS(int64_t, ne,  node, get_elements)
    NNML_TENSOR_LOCALS(size_t,  nb,  node, get_stride_bytes)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    NNML_ASSERT(ne0 == DV);
    NNML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    NNML_ASSERT(nbq0 == nnml_type_size(q->get_data_type()));
    NNML_ASSERT(nbk0 == nnml_type_size(k->get_data_type()));
    NNML_ASSERT(nbv0 == nnml_type_size(v->get_data_type()));

    NNML_ASSERT(neq0 == DK);
    NNML_ASSERT(nek0 == DK);
    NNML_ASSERT(nev0 == DV);

    NNML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    NNML_ASSERT(nb0 == sizeof(float));
    NNML_ASSERT(nb0 <= nb1);
    NNML_ASSERT(nb1 <= nb2);
    NNML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // parallelize by q rows using nnml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) node->get_operation_params() + 0, sizeof(float));
    memcpy(&max_bias,      (float *) node->get_operation_params() + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) node->get_operation_params() + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    nnml_type         const k_vec_dot_type = nnml_get_type_traits(k->get_data_type())->vec_dot_type;
    // printf("k_vec_dot_type: %d\n", k_vec_dot_type);
    nnml_from_float_t const q_to_vec_dot   = nnml_get_type_traits(k_vec_dot_type)->from_float_cpu;
    nnml_vec_dot_t    const kq_vec_dot     = nnml_get_type_traits(k->get_data_type())->vec_dot;
    nnml_to_float_t   const v_to_float     = nnml_get_type_traits(v->get_data_type())->to_float;

    NNML_ASSERT((                                       q_to_vec_dot) && "fattn: unsupported K-type");
    NNML_ASSERT((v->get_data_type() == NNML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    // loop over n_batch and n_head
    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->work_data + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        nnml_fp16_t * VKQ16 = (nnml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        nnml_fp16_t * Q_q   = (nnml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        if (v->get_data_type() == NNML_TYPE_F16) {
            memset(VKQ16, 0, DV*sizeof(nnml_fp16_t));
        } else {
            memset(VKQ32, 0, DV*sizeof(float));
        }

        const nnml_fp16_t * mp = mask ? (nnml_fp16_t *)((char *) mask->tensor_data() + iq1*mask->get_stride_bytes(1) + 
                                    (iq2%mask->get_elements(2))*mask->get_stride_bytes(2) + (iq3%mask->get_elements(3))*mask->get_stride_bytes(3)) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->tensor_data() + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*NNML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->tensor_data() + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->tensor_data() + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->get_data_type() == NNML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    nnml_vec_scale_f16(DV, VKQ16, ms);
                    // printf("New max M: %f, scale ms: %f\n", M, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                    // printf("Old max M: %f, vs: %f\n", M, vs);
                }

                // V += v*expf(s - M)
                nnml_vec_mad_f16(DV, VKQ16, (const nnml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    nnml_vec_scale_f32(DV, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                if (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    nnml_vec_mad_f32(DV, VKQ32, V32, vs);
                } else {
                    // V is F32
                    nnml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->get_data_type() == NNML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = NNML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks
        if (sinks) {
            const float s = ((float *)((char *) sinks->tensor_data()))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                nnml_vec_scale_f32(DV, VKQ32, ms);
            } else {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        // V /= S
        const float S_inv = 1.0f/S;
        nnml_vec_scale_f32(DV, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        // memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // permute(0, 2, 1, 3)
        memcpy((char *) node->tensor_data() + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
}

void nnml_vec_add_f32(float * dst, const float ** srcs, int n_srcs, size_t n) {
    size_t i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t v_acc = vld1q_f32(srcs[0] + i); // load first source vector
        for (int j = 1; j < n_srcs; ++j) {
            float32x4_t v_src = vld1q_f32(srcs[j] + i);
            v_acc = vaddq_f32(v_acc, v_src);
        }
        vst1q_f32(dst + i, v_acc);
    }
    // deal with remaining elements
    for (; i < n; ++i) {
        float sum = srcs[0][i];
        for (int j = 1; j < n_srcs; ++j) sum += srcs[j][i];
        dst[i] = sum;
    }
}

void nnml_vec_add_f16(__fp16 * dst, const __fp16 ** srcs, int n_srcs, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;
    for (; i <= n - 8; i += 8) {
        float16x8_t v_acc = vld1q_f16(srcs[0] + i); // 每次处理 8 个 f16
        for (int j = 1; j < n_srcs; ++j) {
            float16x8_t v_src = vld1q_f16(srcs[j] + i);
            v_acc = vaddq_f16(v_acc, v_src);
        }
        vst1q_f16(dst + i, v_acc);
    }
    for (; i < n; ++i) {
        __fp16 sum = srcs[0][i];
        for (int j = 1; j < n_srcs; ++j) sum += srcs[j][i];
        dst[i] = sum;
    }
#else
    // generic implementation
    for (size_t i = 0; i < n; ++i) {
        float sum = (float)srcs[0][i];
        for (int j = 1; j < n_srcs; ++j) sum += (float)srcs[j][i];
        dst[i] = (__fp16)sum;
    }
#endif
}
