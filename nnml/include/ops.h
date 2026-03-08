/**
 * @file ops.h
 * @brief definitions of operations and related utilities for NNML.
 * 
 * Provides:
 *  - All operation types used in computation graph nodes.
 *  - Macros for extracting tensor properties into local variables for operation implementations.
 *  - Other utilities related to operations.
 * 
 * Designed for collecting all operation-related interfaces in one place.
 */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cmath>

#include "nnml.h"
#include "thread.h"


#define GROUP_MAX_EPS 1e-15f


// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
// example:
//    NNML_TENSOR_LOCALS(int64_t, ne1, src1, get_elements);
//    NNML_TENSOR_LOCALS(size_t,  nb1, src1, get_stride_bytes);
//
#define NNML_TENSOR_LOCALS_1(type, prefix, pointer, func)            \
    const type prefix##0 = (pointer) ? (pointer)->func(0) : 0;       \
    NNML_UNUSED(prefix##0);

#define NNML_TENSOR_LOCALS_2(type, prefix, pointer, func)            \
    NNML_TENSOR_LOCALS_1(type, prefix, pointer, func)                \
    const type prefix##1 = (pointer) ? (pointer)->func(1) : 0;       \
    NNML_UNUSED(prefix##1);

#define NNML_TENSOR_LOCALS_3(type, prefix, pointer, func)            \
    NNML_TENSOR_LOCALS_2(type, prefix, pointer, func)                \
    const type prefix##2 = (pointer) ? (pointer)->func(2) : 0;       \
    NNML_UNUSED(prefix##2);

#define NNML_TENSOR_LOCALS(type, prefix, pointer, func)              \
    NNML_TENSOR_LOCALS_3(type, prefix, pointer, func)                \
    const type prefix##3 = (pointer) ? (pointer)->func(3) : 0;       \
    NNML_UNUSED(prefix##3);

#define NNML_TENSOR_UNARY_OP_LOCALS \
    NNML_TENSOR_LOCALS(int64_t, ne0, src0, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb0, src0, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne,  node,  get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb,  node,  get_stride_bytes)

#define NNML_TENSOR_BINARY_OP_LOCALS \
    NNML_TENSOR_LOCALS(int64_t, ne0, src0, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb0, src0, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne1, src1, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb1, src1, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne,  node,  get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb,  node,  get_stride_bytes)

#define NNML_TENSOR_TERNARY_OP_LOCALS \
    NNML_TENSOR_LOCALS(int64_t, ne0, src0, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb0, src0, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne1, src1, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb1, src1, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne2, src2, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb2, src2, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne,  node,  get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb,  node,  get_stride_bytes)

#define NNML_TENSOR_BINARY_OP_LOCALS01 \
    NNML_TENSOR_LOCALS(int64_t, ne0, src0, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb0, src0, get_stride_bytes) \
    NNML_TENSOR_LOCALS(int64_t, ne1, src1, get_elements) \
    NNML_TENSOR_LOCALS(size_t,  nb1, src1, get_stride_bytes)


enum nnml_op {
    NNML_OP_NONE        = 0,
    NNML_OP_DUP         = 1,
    NNML_OP_ADD         = 2,
    NNML_OP_SUB         = 3,
    NNML_OP_MUL         = 4,
    NNML_OP_MUL_MAT     = 5,
    NNML_OP_DIV         = 6,
    NNML_OP_SQR         = 7,
    NNML_OP_SQRT        = 8,
    NNML_OP_LOG         = 9,
    NNML_OP_SCALE       = 10,
    NNML_OP_PAD         = 11,
    NNML_OP_GET_ROWS    = 12,
    NNML_OP_SET_ROWS    = 13,
    NNML_OP_CPY         = 14,
    NNML_OP_NORM        = 15,
    NNML_OP_RMS_NORM    = 16,
    NNML_OP_RMS_NORM_BACK = 17,
    NNML_OP_GROUP_NORM  = 18,
    NNML_OP_RESHAPE     = 19,
    NNML_OP_VIEW        = 20,
    NNML_OP_PERMUTE     = 21,
    NNML_OP_TRANSPOSE   = 22,
    NNML_OP_ROPE        = 23,
    NNML_OP_UNARY       = 24,
    NNML_OP_GLU         = 25,
    NNML_OP_SOFT_MAX    = 26,
    NNML_OP_CONT        = 27,
    NNML_OP_FLASH_ATTN_EXT = 28,
    NNML_OP_SCATTER_PRE = 29,
    NNML_OP_SCATTER     = 30,
    NNML_OP_GATHER      = 31,
    NNML_OP_COUNT       = 32
};


enum nnml_unary_op {
    NNML_UNARY_OP_ABS,
    NNML_UNARY_OP_SGN,
    NNML_UNARY_OP_ELU,
    NNML_UNARY_OP_RELU,
    NNML_UNARY_OP_SIGMOID,
    NNML_UNARY_OP_GELU,
    NNML_UNARY_OP_SILU,
    NNML_UNARY_OP_TANH,
    NNML_UNARY_OP_EXP,
    NNML_UNARY_OP_COUNT,
};


enum nnml_glu_op {
    NNML_GLU_OP_REGLU,
    NNML_GLU_OP_GEGLU,
    NNML_GLU_OP_SWIGLU,
    NNML_GLU_OP_COUNT,
};


enum nnml_type {
    NNML_TYPE_F32     = 0,
    NNML_TYPE_F16     = 1,
    NNML_TYPE_Q4_0    = 2,
    NNML_TYPE_PH_3    = 3,
    NNML_TYPE_PH_4    = 4,
    NNML_TYPE_PH_5    = 5,
    NNML_TYPE_PH_6    = 6,
    NNML_TYPE_PH_7    = 7,
    NNML_TYPE_Q8_0    = 8,
    NNML_TYPE_PH_9    = 9,
    NNML_TYPE_PH_10   = 10,
    NNML_TYPE_PH_11   = 11,
    NNML_TYPE_PH_12   = 12,
    NNML_TYPE_PH_13   = 13,
    NNML_TYPE_Q6_K    = 14,
    NNML_TYPE_Q8_K    = 15,
    NNML_TYPE_PH_16   = 16,
    NNML_TYPE_PH_17   = 17,
    NNML_TYPE_PH_18   = 18,
    NNML_TYPE_PH_19   = 19,
    NNML_TYPE_PH_20   = 20,
    NNML_TYPE_PH_21   = 21,
    NNML_TYPE_PH_22   = 22,
    NNML_TYPE_PH_23   = 23,
    NNML_TYPE_PH_24   = 24,
    NNML_TYPE_PH_25   = 25,
    NNML_TYPE_I32     = 26,
    NNML_TYPE_I64     = 27,
    NNML_TYPE_F64     = 28,
    NNML_TYPE_COUNT   = 29,
};


enum nnml_prec {
    NNML_PREC_DEFAULT =  0, // stored as nnml_tensor.op_params, 0 by default
    NNML_PREC_F32     = 10,
};


typedef void (*nnml_to_float_t)  (const void  * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);
typedef void (*nnml_from_float_t)(const float * NNML_RESTRICT x, void  * NNML_RESTRICT y, int64_t k);
typedef void (*nnml_vec_dot_t)   (int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT x, size_t bx,
                                  const void * NNML_RESTRICT y, size_t by, int nrc);


/**
 * nnml_type_traits: traits of each nnml_type, used for dispatching to different implementations based on the type of tensors.
 */
struct nnml_type_traits {
    const char *           type_name;
    int64_t                blck_size;
    int64_t                blck_size_interleave; // interleave elements in blocks
    size_t                 type_size;
    bool                   is_quantized;
    nnml_to_float_t        to_float;
    nnml_from_float_t      from_float_ref;

    nnml_from_float_t      from_float_cpu;
    nnml_vec_dot_t         vec_dot;
    nnml_type              vec_dot_type;
    int64_t                nrows;
};
extern const nnml_type_traits type_traits[NNML_TYPE_COUNT];

// type specific helper functions
int64_t                  nnml_blck_size(nnml_type type);
size_t                   nnml_type_size(nnml_type type);
const char *             nnml_type_name(nnml_type type);
bool                     nnml_is_quantized(nnml_type type);
const nnml_type_traits * nnml_get_type_traits(nnml_type type);


// repack the weight tensor to the format suitable for computation, used for quantized format.
int format_repack(nnml_tensor * weight_tensor);
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> int repack(struct nnml_tensor *, const void *, size_t);


// type related functions
template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    nnml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

static_assert(sizeof(block<4, 4>) == 4 * sizeof(nnml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(nnml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(nnml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(nnml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float nnml_compute_fp16_to_fp32(nnml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
    (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline nnml_fp16_t nnml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define NNML_COMPUTE_FP16_TO_FP32(x) nnml_compute_fp16_to_fp32(x)
#define NNML_COMPUTE_FP32_TO_FP16(x) nnml_compute_fp32_to_fp16(x)
#define NNML_FP16_TO_FP32(x) NNML_COMPUTE_FP16_TO_FP32(x)
#define NNML_FP32_TO_FP16(x) NNML_COMPUTE_FP32_TO_FP16(x)

// type functions declarations
typedef double nnml_float;
void nnml_fp16_to_fp32_row(const nnml_fp16_t * x, float * y, int64_t n);
void nnml_fp32_to_fp16_row(const float * x, nnml_fp16_t * y, int64_t n);
void nnml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n);
void nnml_vec_dot_f32(int n, float * NNML_RESTRICT s, size_t bs, const float * NNML_RESTRICT x, size_t bx, const float * NNML_RESTRICT y, size_t by, int nrc);
void nnml_cpu_fp32_to_fp16(const float * x, nnml_fp16_t * y, int64_t n);
void nnml_vec_dot_f16(int n, float * NNML_RESTRICT s, size_t bs, nnml_fp16_t * NNML_RESTRICT x, size_t bx, nnml_fp16_t * NNML_RESTRICT y, size_t by, int nrc);

void dequantize_row_q4_0(const block_q4_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);
void quantize_row_q4_0_ref(const float * NNML_RESTRICT x, block_q4_0 * NNML_RESTRICT y, int64_t k);
void quantize_row_q4_0(const float * NNML_RESTRICT x, void * NNML_RESTRICT y, int64_t k);
void nnml_vec_dot_q4_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc);
void dequantize_row_q6_K(const block_q6_K * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);
void quantize_row_q6_K_ref(const float * NNML_RESTRICT x, block_q6_K * NNML_RESTRICT y, int64_t k);
void nnml_vec_dot_q6_K_q8_K(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc);
void dequantize_row_q8_0(const block_q8_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);
void quantize_row_q8_0_ref(const float * NNML_RESTRICT x, block_q8_0 * NNML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * NNML_RESTRICT x, void * NNML_RESTRICT vy, int64_t k);
void nnml_vec_dot_q8_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc);
void quantize_row_q8_K(const float * NNML_RESTRICT x, void * NNML_RESTRICT y, int64_t k);

void nnml_fp16_to_fp32(const nnml_fp16_t * x, float * y, int64_t n);
void nnml_cpu_fp16_to_fp32(const nnml_fp16_t * x, float * y, int64_t n);

void nnml_vec_scale_f32(const int n, float * y, const float v);
void nnml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);
void nnml_vec_swiglu_f16(const int n, nnml_fp16_t * y, const nnml_fp16_t * x, const nnml_fp16_t * g);


// vector functions declarations
inline static void nnml_vec_cpy_f32 (const int n, float * y, const float * x)               { for (int i = 0; i < n; ++i) y[i]  = x[i]; }
inline static void nnml_vec_acc_f32 (const int n, float * y, const float * x)               { for (int i = 0; i < n; ++i) y[i] += x[i]; }
void nnml_vec_max_f32(const int n, float * s, const float * x);
void nnml_vec_add_f32(float * dst, const float ** srcs, int n_srcs, size_t n);
void nnml_vec_add_f16(__fp16 * dst, const __fp16 ** srcs, int n_srcs, size_t n);


// type conversion table

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
float f16_to_f32(nnml_fp16_t x);
nnml_fp16_t f32_to_f16(float x);

static inline float i32_to_f32(int32_t x) {
    return x;
}

static inline int32_t f32_to_i32(float x) {
    return x;
}

static inline float f32_to_f32(float x) {
    return x;
}

template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<nnml_fp16_t> {
    static constexpr float (*to_f32)(nnml_fp16_t) = f16_to_f32;
    static constexpr nnml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

std::pair<int64_t, int64_t> get_thread_range(nnml_tensor * node, const nnml_compute_state * params);


// entry of all operations

// entry of the forward computation in nnml_compute_node
void nnml_compute_forward_dup(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_add(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_mul(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_mul_mat(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_get_rows(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_set_rows(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_cpy(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_norm(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_rms_norm(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_group_norm(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_reshape(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_view(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_permute(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_transpose(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_rope(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_unary(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_glu(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_soft_max(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_cont(nnml_tensor * node, const nnml_compute_state * params);
void nnml_compute_forward_flash_attn_ext(nnml_tensor * node, const nnml_compute_state * params);

// binary operations
void nnml_binary_compute_forward_add_non_quantized(nnml_tensor * node, const nnml_compute_state * params);
void nnml_binary_compute_forward_sub(nnml_tensor * node, const nnml_compute_state * params);
void nnml_binary_compute_forward_mul(nnml_tensor * node, const nnml_compute_state * params);
void nnml_binary_compute_forward_div(nnml_tensor * node, const nnml_compute_state * params);


// gemm & gemv operations
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, nnml_type PARAM_TYPE>
void forward_mul_mat(nnml_tensor * node, const nnml_compute_state * params);
void forward_mul_mat_generic(nnml_tensor * node, const nnml_compute_state * params);

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, nnml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, nnml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <int64_t INTER_SIZE, nnml_type PARAM_TYPE>
void nnml_quantize_mat_t(const float * NNML_RESTRICT x, void * NNML_RESTRICT vy, int64_t nrow, int64_t n_per_row);


// rope operations
void nnml_compute_forward_rope_f16(nnml_tensor * node, const nnml_compute_state * params);
void nnml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);
void nnml_rope_cache_init(float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale);
void nnml_mrope_cache_init(float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale);


// flash attn ext
void nnml_compute_forward_flash_attn_ext_f16(nnml_tensor * node, const nnml_compute_state * params);


// softmax
void nnml_compute_forward_soft_max_f32(nnml_tensor * node, const nnml_compute_state * params);


// scatter & gather
void nnml_compute_forward_scatter_pre(nnml_tensor * node, nnml_compute_state * params);
void nnml_compute_forward_scatter(nnml_tensor * node, nnml_compute_state * params);
void nnml_compute_forward_gather(nnml_tensor * node, nnml_compute_state * params);
