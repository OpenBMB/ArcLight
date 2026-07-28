#include <cstdio>
#include <cmath>
#include "ops.h"
#include "nnml.h"

static int g_fails = 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include "../src/ops/arm/tq2_i2s.h"
static void check_i2s_cache(void) {
    // 4 output rows x 256-element row (one TQ2_0 block per row). All weights +1 (qs=0xAA, d=1.0).
    const int out_dim = 4, in_dim = QK_K;
    block_tq2_0 w[4];
    for (int r = 0; r < 4; ++r) {
        w[r].d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) w[r].qs[i] = 0xAA;
    }
    tq2_i2s_cache * c = tq2_i2s_build_cache(w, out_dim, in_dim);
    NNML_ASSERT(c != nullptr);
    // packed_size for 1 group, in_dim/64 = 4 sub-blocks -> 4*64 = 256 bytes
    NNML_ASSERT(tq2_i2s_packed_size(out_dim, in_dim) == 256);
    // Each row's scale d == 1.0
    NNML_ASSERT(fabsf(c->scales[0] - 1.0f) < 1e-6f);
    tq2_i2s_free_cache(c);
}

// I2S NEON matmul must match the scalar nnml_vec_dot_tq2_0_q8_K oracle per row.
static void check_i2s_vs_scalar(void) {
    // 4 output rows, in_dim = QK_K. Mixed ternary codes per row.
    const int out_dim = 4, in_dim = QK_K;
    block_tq2_0 w[4];
    // row0 all +1 (0xAA), row1 all -1 (0x00), row2 all 0 (0x55 -> code 1), row3 mixed.
    for (int i = 0; i < QK_K / 4; ++i) {
        w[0].qs[i] = 0xAA;
        w[1].qs[i] = 0x00;
        w[2].qs[i] = 0x55;
        w[3].qs[i] = 0x96;
    }
    for (int r = 0; r < 4; ++r) w[r].d = NNML_FP32_TO_FP16(1.0f);
    tq2_i2s_cache * c = tq2_i2s_build_cache(w, out_dim, in_dim);
    NNML_ASSERT(c != nullptr);

    // Activation = all 1.0.
    float act[QK_K];
    for (int i = 0; i < QK_K; ++i) act[i] = 1.0f;
    int8_t qvec[QK_K];
    float vscale = 0.0f;
    int32_t bsums[QK_K / 256];
    tq2_quantize_vec_i8(act, QK_K, qvec, &vscale, bsums);

    float out_i2s[4] = {0, 0, 0, 0};
    tq2_matmul_i2s_neon(c, qvec, vscale, bsums, 0, out_dim, out_i2s);

    // Scalar oracle: nnml_vec_dot_tq2_0_q8_K expects a block_q8_K activation.
    block_q8_K y;
    quantize_row_q8_K(act, &y, QK_K);
    for (int r = 0; r < 4; ++r) {
        float s_scalar = 0.0f;
        nnml_vec_dot_tq2_0_q8_K(QK_K, &s_scalar, 0, &w[r], 0, &y, 0, 1);
        // Allow tolerance for int8 quant rounding (5% relative).
        if (fabsf(out_i2s[r] - s_scalar) > 0.05f * (fabsf(s_scalar) + 1.0f)) {
            printf("FAIL i2s row %d: i2s=%g scalar=%g\n", r, out_i2s[r], s_scalar);
            ++g_fails;
        }
    }
    tq2_i2s_free_cache(c);
}
#endif

static void check(float got, float expv, const char * msg) {
    if (fabsf(got - expv) > 1e-5f) {
        printf("FAIL %s: got %g, expected %g\n", msg, got, expv);
        ++g_fails;
    }
}

// Q4_K NEON vec_dot must match the scalar _ref oracle within relative tolerance.
// Builds one Q4_K block (256 weights) + a Q8_K activation from a known f32 vector,
// then compares nnml_vec_dot_q4_K_q8_K (NEON) vs nnml_vec_dot_q4_K_q8_K_ref (scalar).
static void check_q4k_vecdot_single(void) {
    block_q4_K w; memset(&w, 0, sizeof(w));
    w.d    = NNML_FP32_TO_FP16(1.0f);
    w.dmin = NNML_FP32_TO_FP16(0.1f);
    // 8 sub-block scales/mins via get_scale_min_k4: varied 6-bit sc/m values.
    for (int i = 0; i < 12; ++i) w.scales[i] = (i % 2) ? 0x21 : 0x12;
    // qs nibbles: a repeating pattern (each byte -> lo + hi nibble in 0..15).
    for (int i = 0; i < QK_K / 2; ++i) w.qs[i] = (uint8_t)((i * 7) & 0xFF);

    float act[QK_K];
    for (int i = 0; i < QK_K; ++i) act[i] = 0.5f * (float)((i % 7) - 3);  // small varied values
    block_q8_K y; quantize_row_q8_K(act, &y, QK_K);

    float s_neon = 0.0f, s_ref = 0.0f;
    nnml_vec_dot_q4_K_q8_K    (QK_K, &s_neon, 0, &w, 0, &y, 0, 1);
    nnml_vec_dot_q4_K_q8_K_ref(QK_K, &s_ref,  0, &w, 0, &y, 0, 1);
    float tol = 1e-3f * (fabsf(s_ref) + 1.0f);
    if (fabsf(s_neon - s_ref) > tol) {
        printf("FAIL q4k vecdot single: neon=%g ref=%g (tol=%g)\n", s_neon, s_ref, tol);
        ++g_fails;
    }
}

// Multi-block, varied data: pseudo-random fill across several blocks and activations.
static void check_q4k_vecdot_multi(void) {
    constexpr int nb = 5;
    const int n  = nb * QK_K;
    block_q4_K w[nb];
    block_q8_K y[nb];
    uint32_t rng = 0x1234abcdu;
    auto next_u32 = [&]() -> uint32_t { rng = rng * 1664525u + 1013904223u; return rng; };

    for (int b = 0; b < nb; ++b) {
        memset(&w[b], 0, sizeof(w[b]));
        w[b].d    = NNML_FP32_TO_FP16(0.5f + (float)(next_u32() & 0xff) / 255.0f);
        w[b].dmin = NNML_FP32_TO_FP16(0.05f + (float)(next_u32() & 0xff) / 2550.0f);
        for (int i = 0; i < 12; ++i) w[b].scales[i] = (uint8_t)(next_u32() & 0xff);
        for (int i = 0; i < QK_K / 2; ++i) w[b].qs[i] = (uint8_t)(next_u32() & 0xff);

        float act[QK_K];
        for (int i = 0; i < QK_K; ++i) act[i] = ((float)(int32_t)(next_u32() & 0xf) - 7.0f) * 0.1f;
        quantize_row_q8_K(act, &y[b], QK_K);
    }

    float s_neon = 0.0f, s_ref = 0.0f;
    nnml_vec_dot_q4_K_q8_K    (n, &s_neon, 0, w, 0, y, 0, 1);
    nnml_vec_dot_q4_K_q8_K_ref(n, &s_ref,  0, w, 0, y, 0, 1);
    float tol = 1e-3f * (fabsf(s_ref) + 1.0f);
    if (fabsf(s_neon - s_ref) > tol) {
        printf("FAIL q4k vecdot multi: neon=%g ref=%g (tol=%g)\n", s_neon, s_ref, tol);
        ++g_fails;
    }
}

int main(void) {
    // Block A: scale=1.0, all packed bytes zero -> every 2-bit code=0 -> output (0-1)*1.0 = -1.0
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        for (int i = 0; i < QK_K; ++i) check(y[i], -1.0f, "blockA all -1*d");
    }
    // Block B: scale=1.0, qs[0]=0xFF (all four 2-bit codes of byte 0 == 3), rest zero.
    // Dequant order within the first 32-byte group (j=0) emits 4 planes l=0..3 at output
    // offsets 0/32/64/96; for m=0 (qs[0]) every plane reads code 3 -> (3-1)*1.0 = 2.0.
    // m>=1 (qs[1..31]==0) -> -1.0. Second 32-byte group (j=32) all zero -> -1.0.
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0;
        b.qs[0] = 0xFF;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        check(y[0],    2.0f, "blockB y[0]=2");
        check(y[32],   2.0f, "blockB y[32]=2");
        check(y[64],   2.0f, "blockB y[64]=2");
        check(y[96],   2.0f, "blockB y[96]=2");
        check(y[1],   -1.0f, "blockB y[1]=-1");
        check(y[128], -1.0f, "blockB y[128]=-1 (group 2)");
        check(y[255], -1.0f, "blockB y[255]=-1");
    }
    // Block C: scale=2.0, qs all 0xAA (low two bits = 0b10 = 2) -> (2-1)*2.0 = 2.0 everywhere
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(2.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0xAA;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        for (int i = 0; i < QK_K; ++i) check(y[i], 2.0f, "blockC all (2-1)*2");
    }

    // vec_dot regression: weights all +1 (qs=0xAA -> code 2 -> (2-1)*1.0 = +1.0)
    // dotted with an all-ones activation quantized to Q8_K must equal 256.0.
    // Catches a vec_dot_type / activation block-size mismatch: Q8_0 blocks hold
    // 32 elements but this vec_dot indexes 256 per block, so vec_dot_type MUST be Q8_K.
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0xAA;   // every 2-bit code = 2 -> weight +1.0
        float act[QK_K];
        for (int i = 0; i < QK_K; ++i) act[i] = 1.0f;
        block_q8_K y;
        quantize_row_q8_K(act, &y, QK_K);
        float s = 0.0f;
        nnml_vec_dot_tq2_0_q8_K(QK_K, &s, 0, &b, 0, &y, 0, 1);
        check(s, 256.0f, "vec_dot all-ones = 256");
    }
    // weights all -1 (qs=0x00 -> code 0 -> (0-1)*1.0 = -1.0) -> dot = -256.0
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0x00;   // every 2-bit code = 0 -> weight -1.0
        float act[QK_K];
        for (int i = 0; i < QK_K; ++i) act[i] = 1.0f;
        block_q8_K y;
        quantize_row_q8_K(act, &y, QK_K);
        float s = 0.0f;
        nnml_vec_dot_tq2_0_q8_K(QK_K, &s, 0, &b, 0, &y, 0, 1);
        check(s, -256.0f, "vec_dot all-neg-ones = -256");
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    check_i2s_cache();
    check_i2s_vs_scalar();
#endif

    check_q4k_vecdot_single();
    check_q4k_vecdot_multi();

    if (g_fails == 0) { printf("test-tq2-0: PASS\n"); return 0; }
    printf("test-tq2-0: FAIL (%d assertions)\n", g_fails);
    return 1;
}
