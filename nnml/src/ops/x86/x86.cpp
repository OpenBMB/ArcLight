#include <cassert>

#include <cmath>

#include "ops.h"
#include "tensor.h"


#if !defined(NNML_CPU_FP32_TO_FP16)
#define NNML_CPU_FP32_TO_FP16(x) NNML_COMPUTE_FP32_TO_FP16(x)
#endif

#if defined(__F16C__)
    #ifdef _MSC_VER
        #define NNML_CPU_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #define NNML_CPU_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
    #else
        #define NNML_CPU_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
        #define NNML_CPU_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
    #endif

    #define NNML_CPU_FP16_TO_FP32(x) NNML_CPU_COMPUTE_FP16_TO_FP32(x)
#endif

#if defined(__AVX512F__)
    #define NNML_F32_STEP 64
    #define NNML_F32_EPR  16
    #define NNML_F32x16         __m512
    #define NNML_F32_VEC        NNML_F32x16
    #define NNML_F32x16_ZERO    _mm512_setzero_ps()
    #define NNML_F32_VEC_ZERO   NNML_F32x16_ZERO
    #define NNML_F32x16_LOAD    _mm512_loadu_ps
    #define NNML_F32_VEC_LOAD   NNML_F32x16_LOAD
    #define NNML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
    #define NNML_F32_VEC_FMA    NNML_F32x16_FMA
    #define NNML_F32x16_REDUCE(res, x)                                    \
    do {                                                                  \
        int offset = NNML_F32_ARR >> 1;                                   \
        for (int i = 0; i < offset; ++i) {                                \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
        }                                                                 \
        offset >>= 1;                                                     \
        for (int i = 0; i < offset; ++i) {                                \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
        }                                                                 \
        offset >>= 1;                                                     \
        for (int i = 0; i < offset; ++i) {                                \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
        }                                                                 \
        res = (nnml_float) _mm512_reduce_add_ps(x[0]);                    \
    } while (0)
    #define NNML_F32_VEC_REDUCE NNML_F32x16_REDUCE
    #define NNML_F32x16_SET1(x) _mm512_set1_ps(x)
    #define NNML_F32_VEC_SET1   NNML_F32x16_SET1
    #define NNML_F32x16_MUL     _mm512_mul_ps
    #define NNML_F32_VEC_MUL    NNML_F32x16_MUL
    #define NNML_F32x16_STORE   _mm512_storeu_ps
    #define NNML_F32_VEC_STORE  NNML_F32x16_STORE

    #define NNML_F16_STEP 64
    #define NNML_F16_EPR  16
    #define NNML_F32Cx16             __m512
    #define NNML_F16_VEC             NNML_F32Cx16
    #define NNML_F32Cx16_MUL         _mm512_mul_ps
    #define NNML_F16_VEC_MUL         NNML_F32Cx16_MUL
    #define NNML_F32Cx16_SET1(x)     _mm512_set1_ps(x)
    #define NNML_F16_VEC_SET1        NNML_F32Cx16_SET1
    #define NNML_F32Cx16_ZERO        _mm512_setzero_ps()
    #define NNML_F16_VEC_ZERO        NNML_F32Cx16_ZERO
    #define NNML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
    #define NNML_F16_VEC_LOAD(p, i)  NNML_F32Cx16_LOAD(p)
    #define NNML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))
    #define NNML_F16_VEC_STORE(p, r, i) NNML_F32Cx16_STORE(p, r[i])
    #define NNML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
    #define NNML_F16_VEC_FMA          NNML_F32Cx16_FMA
    #define NNML_F32Cx16_REDUCE(res, x)                               \
    do {                                                              \
        int offset = NNML_F32_ARR >> 1;                               \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        res = (nnml_float) _mm512_reduce_add_ps(x[0]);                \
    } while (0)
    #define NNML_F16_VEC_REDUCE       NNML_F32Cx16_REDUCE

#elif defined(__AVX__)
    #define NNML_F32_STEP 32
    #define NNML_F32_EPR  8
    #define NNML_F32x8          __m256
    #define NNML_F32_VEC        NNML_F32x8
    #define NNML_F32x8_ZERO     _mm256_setzero_ps()
    #define NNML_F32_VEC_ZERO   NNML_F32x8_ZERO
    #define NNML_F32x8_LOAD     _mm256_loadu_ps
    #define NNML_F32_VEC_LOAD   NNML_F32x8_LOAD
    #define NNML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
    #define NNML_F32_VEC_FMA    NNML_F32x8_FMA
    #define NNML_F32x8_REDUCE(res, x)                                 \
    do {                                                              \
        int offset = NNML_F32_ARR >> 1;                               \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
        }                                                             \
        const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                    _mm256_extractf128_ps(x[0], 1)); \
        const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
        res = (nnml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
    } while (0)
    #define NNML_F32_VEC_REDUCE NNML_F32x8_REDUCE
    #define NNML_F32x8_SET1(x) _mm256_set1_ps(x)
    #define NNML_F32_VEC_SET1   NNML_F32x8_SET1
    #define NNML_F32x8_MUL     _mm256_mul_ps
    #define NNML_F32_VEC_MUL    NNML_F32x8_MUL
    #define NNML_F32x8_STORE   _mm256_storeu_ps
    #define NNML_F32_VEC_STORE  NNML_F32x8_STORE

    #define NNML_F16_STEP 32
    #define NNML_F16_EPR  8
    #define NNML_F32Cx8             __m256
    #define NNML_F16_VEC            NNML_F32Cx8
    #define NNML_F32Cx8_MUL         _mm256_mul_ps
    #define NNML_F16_VEC_MUL        NNML_F32Cx8_MUL
    #define NNML_F32Cx8_SET1(x)     _mm256_set1_ps(x)
    #define NNML_F16_VEC_SET1       NNML_F32Cx8_SET1
    #define NNML_F32Cx8_ZERO        _mm256_setzero_ps()
    #define NNML_F16_VEC_ZERO       NNML_F32Cx8_ZERO
    #define NNML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
    #define NNML_F16_VEC_LOAD(p, i) NNML_F32Cx8_LOAD(p)
    #define NNML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
    #define NNML_F16_VEC_STORE(p, r, i) NNML_F32Cx8_STORE(p, r[i])
    #define NNML_F32Cx8_FMA         NNML_F32x8_FMA
    #define NNML_F16_VEC_FMA        NNML_F32Cx8_FMA
    #define NNML_F32Cx8_REDUCE      NNML_F32x8_REDUCE
    #define NNML_F16_VEC_REDUCE     NNML_F32Cx8_REDUCE

#else
    ERROR "No SIMD support found"
#endif

#define NNML_F32_ARR (NNML_F32_STEP/NNML_F32_EPR)
#define NNML_F16_ARR (NNML_F16_STEP/NNML_F16_EPR)

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                           const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) {
    const __m128i mone = _mm_set1_epi16(1);

    const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
    const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
    // NNML_CPU_FP16_TO_FP32 is faster than Intel F16C
    return _mm256_set_m128(_mm_set1_ps(NNML_CPU_FP16_TO_FP32(x1) * NNML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(NNML_CPU_FP16_TO_FP32(x0) * NNML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 quad_mx_delta_float(const int8_t x0, const float y0, const int8_t x1, const float y1) {
    return _mm256_set_m128(_mm_set1_ps(NNML_E8M0_TO_FP32_HALF(x1) * NNML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(NNML_E8M0_TO_FP32_HALF(x0) * NNML_CPU_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)


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

    *s = sumf;
}

void nnml_vec_dot_f16(int n, float * NNML_RESTRICT s, size_t bs, nnml_fp16_t * NNML_RESTRICT x, size_t bx, nnml_fp16_t * NNML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    NNML_UNUSED(nrc);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);
    NNML_UNUSED(bs);

    nnml_float sumf = 0.0;

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
    assert(!isnan(sumf) && !isinf(sumf));

    *s = sumf;
}

#if __AVX__ || __AVX2__ || __AVX512F__
// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

void nnml_vec_dot_q4_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * NNML_RESTRICT x = (const block_q4_0 * NNML_RESTRICT)vx;
    const block_q8_0 * NNML_RESTRICT y = (const block_q8_0 * NNML_RESTRICT)vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( NNML_CPU_FP16_TO_FP32(x[ib].d) * NNML_CPU_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
        const __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
        const __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

        const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
        const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
        const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
        const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
        const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
        const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
        const __m256 p =  sum_i16_pairs_float(p_2, p_1);

        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( NNML_CPU_FP16_TO_FP32(x[ib].d) * NNML_CPU_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( NNML_CPU_FP16_TO_FP32(x[ib + 1].d) * NNML_CPU_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

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
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * NNML_RESTRICT x = (block_q6_K * NNML_RESTRICT)vx;
    const block_q8_K * NNML_RESTRICT y = (block_q8_K * NNML_RESTRICT)vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * NNML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * NNML_RESTRICT q4 = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const int8_t  * NNML_RESTRICT q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * NNML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * NNML_RESTRICT q4 = x[i].ql;
        const uint8_t * NNML_RESTRICT qh = x[i].qh;
        const int8_t  * NNML_RESTRICT q8 = y[i].qs;

        // handle the q6_k -32 offset separately using bsums
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));

        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        const __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    nnml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// x86 SIMD Q4_K vec_dot is out of scope; defer to the portable scalar reference.
void nnml_vec_dot_q4_K_q8_K(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    nnml_vec_dot_q4_K_q8_K_ref(n, s, bs, vx, bx, vy, by, nrc);
}

void nnml_vec_dot_q8_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * NNML_RESTRICT x = (block_q8_0 * NNML_RESTRICT)vx;
    const block_q8_0 * NNML_RESTRICT y = (block_q8_0 * NNML_RESTRICT)vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(NNML_CPU_FP16_TO_FP32(x[ib].d) * NNML_CPU_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i qx_1_1 = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i qx_2_0 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i qx_2_1 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs + 1);
        const __m128i qy_1_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i qy_1_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const __m128i qy_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i qy_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
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
}

inline static void nnml_vec_scale_f16(const int n, nnml_fp16_t * y, const float v) {
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
}

inline static void nnml_vec_mad_f32(const int n, float * NNML_RESTRICT y, const float * NNML_RESTRICT x, const float v) {
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
}

inline static void nnml_vec_mad_f16(const int n, nnml_fp16_t * NNML_RESTRICT y, const nnml_fp16_t * NNML_RESTRICT x, const float v) {
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
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 nnml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 nnml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = nnml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 nnml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 nnml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = nnml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}
#endif

inline static float nnml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

void nnml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(nnml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(nnml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(nnml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
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

#if defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = NNML_CPU_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
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

nnml_float nnml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    nnml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = nnml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (nnml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = nnml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (nnml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = nnml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (nnml_float)_mm_cvtss_f32(val);
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
#if defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        // load first source vector
        __m256 v_acc = _mm256_loadu_ps(srcs[0] + i);

        for (int j = 1; j < n_srcs; ++j) {
            __m256 v_src = _mm256_loadu_ps(srcs[j] + i);
            v_acc = _mm256_add_ps(v_acc, v_src);
        }
        _mm256_storeu_ps(dst + i, v_acc);
    }
    // deal with remaining elements
    for (; i < n; ++i) {
        float sum = srcs[0][i];
        for (int j = 1; j < n_srcs; ++j) sum += srcs[j][i];
        dst[i] = sum;
    }
#else
    ERROR "no SIMD add operation supported!"
#endif
}

void nnml_vec_add_f16(nnml_fp16_t * dst, const nnml_fp16_t ** srcs, int n_srcs, size_t n) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= n; i += 8) {
        // load first source vector
        __m256 v_acc = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(srcs[0] + i)));

        for (int j = 1; j < n_srcs; ++j) {
            __m256 v_src = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(srcs[j] + i)));
            v_acc = _mm256_add_ps(v_acc, v_src);
        }

        _mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(v_acc, _MM_FROUND_TO_NEAREST_INT));
    }
    // deal with remaining elements
    for (; i < n; ++i) {
        float sum = NNML_CPU_FP16_TO_FP32(srcs[0][i]);
        for (int j = 1; j < n_srcs; ++j) sum += NNML_CPU_FP16_TO_FP32(srcs[j][i]);
        dst[i] = NNML_CPU_FP32_TO_FP16(sum);
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
