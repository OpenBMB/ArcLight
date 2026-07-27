#include <cassert>
#include <cmath>
#include <cstring>

#include "ops.h"

int main() {
    block_q4_K q4{};
    q4.d = NNML_FP32_TO_FP16(1.0f);
    q4.dmin = NNML_FP32_TO_FP16(0.0f);

    // Encode scale 1 and minimum 0 for each of the eight 32-value groups.
    for (int i = 0; i < 4; ++i) {
        q4.scales[i] = 1;
        q4.scales[i + 4] = 0;
        q4.scales[i + 8] = 1;
    }
    std::memset(q4.qs, 0x32, sizeof(q4.qs));

    float values[QK_K];
    dequantize_row_q4_K(&q4, values, QK_K);
    for (int group = 0; group < QK_K / 64; ++group) {
        for (int i = 0; i < 32; ++i) {
            assert(values[group * 64 + i] == 2.0f);
            assert(values[group * 64 + 32 + i] == 3.0f);
        }
    }

    block_q8_K q8{};
    q8.d = 0.5f;
    std::memset(q8.qs, 1, sizeof(q8.qs));
    float dot = 0.0f;
    nnml_vec_dot_q4_K_q8_K(QK_K, &dot, 0, &q4, 0, &q8, 0, 1);
    assert(std::fabs(dot - 320.0f) < 1e-6f);

    assert(nnml_blck_size(NNML_TYPE_Q4_K) == QK_K);
    assert(nnml_type_size(NNML_TYPE_Q4_K) == sizeof(block_q4_K));
    return 0;
}
