// Stateful short-convolution primitives for linear-attention / SSM layers.
//
// nnml_ssm_conv1d_update is the depthwise causal Conv1d recurrent update used by
// Qwen3.5's GatedDeltaNet linear-attention layers. It is a portable scalar
// reference (ARM NEON optimization is Phase 2); correctness against the
// HuggingFace torch_causal_conv1d_update reference is covered by test-ssm.cpp.

#include "ssm.h"

#include <cmath>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
static inline float nnml_silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

void nnml_ssm_conv1d_update(const float * x, float * conv_state, const float * weight,
                            int conv_dim, int kernel, float * out) {
    // Degenerate inputs: nothing to do / write. A zero kernel has no taps so the
    // output is undefined; we leave `out` untouched rather than read OOB.
    if (x == nullptr || conv_state == nullptr || weight == nullptr ||
        out == nullptr || conv_dim <= 0 || kernel <= 0) {
        return;
    }

    // conv_state and weight are channel-major: channel c's `kernel` taps are
    // contiguous, so element (tap k, channel c) is at flat offset c*kernel + k.
    // This matches the ggml layout of a [kernel, conv_dim] tensor (ne[0]=kernel,
    // the contiguous axis), the GGUF ssm_conv1d.weight [kernel, conv_dim], and
    // the llm_ssm_state conv_state allocation [kernel, conv_dim].
    for (int c = 0; c < conv_dim; ++c) {
        float *       sc = conv_state + (size_t)c * kernel;
        const float * wc = weight     + (size_t)c * kernel;

        // Roll this channel's taps left by one (drop tap 0) and append the new
        // input x[c] at the last tap. After the update, tap k holds x_{t-kernel+1+k}.
        for (int k = 0; k < kernel - 1; ++k) {
            sc[k] = sc[k + 1];
        }
        sc[kernel - 1] = x[c];

        // Depthwise dot of the rolled state with the conv weights, then SiLU.
        // out[c] = SiLU( sum_k state[k,c] * weight[k,c] ).
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            acc += sc[k] * wc[k];
        }
        out[c] = nnml_silu_f32(acc);
    }
}

void nnml_ssm_conv1d_update_seq(const float * x, size_t x_nb1,
                                float * conv_state, const float * weight,
                                int conv_dim, int kernel, int n_tokens,
                                float * out, size_t out_nb1, int ith, int nth) {
    // Degenerate inputs: nothing to do / write.
    if (x == nullptr || conv_state == nullptr || weight == nullptr ||
        out == nullptr || conv_dim <= 0 || kernel <= 0 || n_tokens <= 0) {
        return;
    }
    // Channels are independent (depthwise) and parallelized across threads via
    // contiguous per-thread ranges (NOT an ith-stride: channels are contiguous in
    // memory, so a stride would false-share cache lines between threads). Tokens are
    // sequential because conv_state rolls left each step.
    const int c0 = (int)((int64_t)ith * conv_dim / nth);
    const int c1 = (int)((int64_t)(ith + 1) * conv_dim / nth);
    for (int c = c0; c < c1; ++c) {
        float *       sc = conv_state + (size_t)c * kernel;   // channel c's `kernel` taps
        const float * wc = weight     + (size_t)c * kernel;
        for (int t = 0; t < n_tokens; ++t) {
            const float xc = *(const float *)((const char *)x + (size_t)t * x_nb1
                                              + (size_t)c * sizeof(float));
            // Roll this channel's taps left by one (drop tap 0), append x[t][c] at the
            // last tap. After the update, tap k holds x_{t-kernel+1+k}.
            for (int k = 0; k < kernel - 1; ++k) {
                sc[k] = sc[k + 1];
            }
            sc[kernel - 1] = xc;
            // Depthwise dot of the rolled state with the conv weights, then SiLU.
            // kernel == 4 and sc/wc are 4 contiguous floats -> one f32x4 mul + reduce.
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (kernel == 4) {
                float32x4_t prod = vmulq_f32(vld1q_f32(sc), vld1q_f32(wc));
                float acc = vaddvq_f32(prod);
                *(float *)((char *)out + (size_t)t * out_nb1 + (size_t)c * sizeof(float)) =
                    nnml_silu_f32(acc);
            } else
#endif
            {
                float acc = 0.0f;
                for (int k = 0; k < kernel; ++k) {
                    acc += sc[k] * wc[k];
                }
                *(float *)((char *)out + (size_t)t * out_nb1 + (size_t)c * sizeof(float)) =
                    nnml_silu_f32(acc);
            }
        }
    }
}

void nnml_ssm_gated_delta_rule_update(const float * q, const float * k, const float * v,
                                      float g, float beta, float * S,
                                      int k_dim, int v_dim, float * out) {
    // Degenerate inputs: nothing to do / write. Leave `out` untouched.
    if (q == nullptr || k == nullptr || v == nullptr || S == nullptr ||
        out == nullptr || k_dim <= 0 || v_dim <= 0) {
        return;
    }

    // 1. L2-normalize q and k (HF l2norm: x * rsqrt(sumsq + eps), eps = 1e-6),
    //    and apply the query scale 1/sqrt(k_dim). We only need the per-vector
    //    inverse-norm scalars: q_n[k] = q[k] * q_inv_norm / sqrt(k_dim) and
    //    k_n[k] = k[k] * k_inv_norm, computed on the fly in the loops below.
    const float eps = 1e-6f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    // NEON path: requires k_dim, v_dim multiples of 4 (true for Qwen3.5: 128) and a
    // small fixed scratch (v_dim <= 128) so we can use a stack buffer, not a heap alloc,
    // in this per-token-per-head primitive.
    if ((k_dim & 3) == 0 && (v_dim & 3) == 0 && v_dim <= 128) {
        // 1. L2-norm sum-of-squares (reduce over k_dim).
        float32x4_t qs = vdupq_n_f32(0.0f), kss = vdupq_n_f32(0.0f);
        for (int i = 0; i < k_dim; i += 4) {
            float32x4_t qv = vld1q_f32(q + i);
            float32x4_t kv = vld1q_f32(k + i);
            qs  = vfmaq_f32(qs,  qv, qv);
            kss = vfmaq_f32(kss, kv, kv);
        }
        const float q_scale   = 1.0f / (std::sqrt((float)vaddvq_f32(qs)  + eps) * std::sqrt((float)k_dim));
        const float k_inv_norm = 1.0f / std::sqrt((float)vaddvq_f32(kss) + eps);

        // 2. Decay: S <- exp(g) * S  (broadcast scale over the k_dim*v_dim block).
        const float32x4_t dv = vdupq_n_f32(std::exp(g));
        const int N = k_dim * v_dim;
        for (int i = 0; i < N; i += 4) vst1q_f32(S + i, vmulq_f32(vld1q_f32(S + i), dv));

        // 3. kv_mem[v] = sum_k S[k,v] * k[k]  (iterate k outer, vectorize over the
        //    contiguous v axis). Block v by 16 (4 f32x4 accumulators kept in registers
        //    across the k loop) so the accumulator is not reloaded each k step.
        //    Stack scratch reused for delta afterwards.
        float kv[128] = {0};
        for (int vv0 = 0; vv0 < v_dim; vv0 += 16) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f),
                         a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            for (int kk = 0; kk < k_dim; ++kk) {
                const float32x4_t kb = vdupq_n_f32(k[kk]);
                const float * sk = S + (size_t)kk * v_dim + vv0;
                a0 = vfmaq_f32(a0, kb, vld1q_f32(sk + 0));
                a1 = vfmaq_f32(a1, kb, vld1q_f32(sk + 4));
                a2 = vfmaq_f32(a2, kb, vld1q_f32(sk + 8));
                a3 = vfmaq_f32(a3, kb, vld1q_f32(sk + 12));
            }
            vst1q_f32(kv + vv0 + 0,  a0);
            vst1q_f32(kv + vv0 + 4,  a1);
            vst1q_f32(kv + vv0 + 8,  a2);
            vst1q_f32(kv + vv0 + 12, a3);
        }
        // delta[v] = (v[v] - kv_mem[v]*k_inv_norm) * beta  (kv[] now holds delta).
        const float32x4_t kin = vdupq_n_f32(k_inv_norm), betav = vdupq_n_f32(beta);
        for (int vv = 0; vv < v_dim; vv += 4) {
            float32x4_t vmem = vmulq_f32(vld1q_f32(kv + vv), kin);
            vst1q_f32(kv + vv, vmulq_f32(vsubq_f32(vld1q_f32(v + vv), vmem), betav));
        }
        // 4. Rank-1 update: S[k,v] += (k[k]*k_inv_norm) * delta[v]  (writes S directly).
        for (int kk = 0; kk < k_dim; ++kk) {
            const float32x4_t kn = vdupq_n_f32(k[kk] * k_inv_norm);
            float * S_k = S + (size_t)kk * v_dim;
            for (int vv = 0; vv < v_dim; vv += 4) {
                vst1q_f32(S_k + vv, vfmaq_f32(vld1q_f32(S_k + vv), kn, vld1q_f32(kv + vv)));
            }
        }
        // 5. Output: out[v] = sum_k S[k,v] * (q[k]*q_scale)  (post-update S), blocked.
        for (int vv0 = 0; vv0 < v_dim; vv0 += 16) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f),
                         a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            for (int kk = 0; kk < k_dim; ++kk) {
                const float32x4_t qn = vdupq_n_f32(q[kk] * q_scale);
                const float * sk = S + (size_t)kk * v_dim + vv0;
                a0 = vfmaq_f32(a0, qn, vld1q_f32(sk + 0));
                a1 = vfmaq_f32(a1, qn, vld1q_f32(sk + 4));
                a2 = vfmaq_f32(a2, qn, vld1q_f32(sk + 8));
                a3 = vfmaq_f32(a3, qn, vld1q_f32(sk + 12));
            }
            vst1q_f32(out + vv0 + 0,  a0);
            vst1q_f32(out + vv0 + 4,  a1);
            vst1q_f32(out + vv0 + 8,  a2);
            vst1q_f32(out + vv0 + 12, a3);
        }
        return;
    }
#endif
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    for (int i = 0; i < k_dim; ++i) {
        q_sumsq += q[i] * q[i];
        k_sumsq += k[i] * k[i];
    }
    const float q_scale = 1.0f / (std::sqrt(q_sumsq + eps) * std::sqrt((float)k_dim));
    const float k_inv_norm = 1.0f / std::sqrt(k_sumsq + eps);

    // 2. Decay: S <- exp(g) * S. (g is the raw scalar; HF applies .exp().)
    const float decay = std::exp(g);
    for (int i = 0; i < k_dim * v_dim; ++i) {
        S[i] *= decay;
    }

    // 3. kv_mem[v] = sum_k S[k,v] * k_n[k], then delta[v] = (v[v] - kv_mem[v]) * beta.
    //    k_n[k] = k[k] * k_inv_norm, so k_inv_norm factors out of the contraction.
    std::vector<float> delta((size_t)v_dim);
    for (int vv = 0; vv < v_dim; ++vv) {
        float kv_mem = 0.0f;
        for (int kk = 0; kk < k_dim; ++kk) {
            kv_mem += S[(size_t)kk * v_dim + vv] * k[kk];
        }
        kv_mem *= k_inv_norm;
        delta[vv] = (v[vv] - kv_mem) * beta;
    }

    // 4. Rank-1 update: S[k,v] += k_n[k] * delta[v]  (outer product).
    for (int kk = 0; kk < k_dim; ++kk) {
        const float kn = k[kk] * k_inv_norm;
        float * S_k = S + (size_t)kk * v_dim;
        for (int vv = 0; vv < v_dim; ++vv) {
            S_k[vv] += kn * delta[vv];
        }
    }

    // 5. Output: out[v] = sum_k S[k,v] * q_n[k]  (contract the k axis).
    //    Computed against the post-update S (matches HF: state updated, then
    //    output read from it). q_n[k] = q[k] * q_scale factors out per k-row.
    for (int vv = 0; vv < v_dim; ++vv) {
        out[vv] = 0.0f;
    }
    for (int kk = 0; kk < k_dim; ++kk) {
        const float qn = q[kk] * q_scale;
        const float * S_k = S + (size_t)kk * v_dim;
        for (int vv = 0; vv < v_dim; ++vv) {
            out[vv] += S_k[vv] * qn;
        }
    }
}

void nnml_ssm_delta_rule_update_seq(const float * q, size_t q_nb2, const float * k, size_t k_nb2,
                                    const float * v, size_t v_nb2, const float * g, size_t g_nb1,
                                    const float * beta, size_t b_nb1, float * S,
                                    int k_dim, int v_dim, int num_heads, int n_tokens,
                                    float * out, size_t out_nb2, int ith, int nth) {
    // Degenerate inputs: nothing to do / write.
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr ||
        beta == nullptr || S == nullptr || out == nullptr ||
        k_dim <= 0 || v_dim <= 0 || num_heads <= 0 || n_tokens <= 0) {
        return;
    }
    // Heads are independent (each owns a k_dim x v_dim recurrent-state block) and
    // parallelized across threads via contiguous per-thread ranges (avoids false
    // sharing). Tokens are sequential because S_{t} depends on S_{t-1}.
    const int h0 = ith * num_heads / nth;
    const int h1 = (ith + 1) * num_heads / nth;
    for (int h = h0; h < h1; ++h) {
        float * Sh = S + (size_t)h * k_dim * v_dim;
        for (int t = 0; t < n_tokens; ++t) {
            const float * qh = (const float *)((const char *)q + (size_t)t * q_nb2) + (size_t)h * k_dim;
            const float * kh = (const float *)((const char *)k + (size_t)t * k_nb2) + (size_t)h * k_dim;
            const float * vh = (const float *)((const char *)v + (size_t)t * v_nb2) + (size_t)h * v_dim;
            const float   gt = *((const float *)((const char *)g + (size_t)t * g_nb1) + h);
            const float   bt = *((const float *)((const char *)beta + (size_t)t * b_nb1) + h);
            float *       oh = (float *)((char *)out + (size_t)t * out_nb2) + (size_t)h * v_dim;
            nnml_ssm_gated_delta_rule_update(qh, kh, vh, gt, bt, Sh, k_dim, v_dim, oh);
        }
    }
}

void nnml_ssm_rms_norm_gated(const float * x, const float * weight, const float * gate,
                             int n, float eps, float * out) {
    // Degenerate inputs: safe no-op (leave `out` untouched).
    if (x == nullptr || weight == nullptr || gate == nullptr ||
        out == nullptr || n <= 0) {
        return;
    }

    // 1. variance = mean(x*x). Accumulate in double to match the numpy
    //    float64 reference and absorb f32 rounding in the sum.
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        sumsq += (double)x[i] * (double)x[i];
    }
    const double rinv = 1.0 / std::sqrt(sumsq / (double)n + (double)eps);

    // 2. h = weight * (x * rsqrt(variance+eps)), then out = h * SiLU(gate).
    //    SiLU(g) = g / (1 + exp(-g)). Computed in double then narrowed to f32.
    for (int i = 0; i < n; ++i) {
        const double g = (double)gate[i];
        const double silu_g = g / (1.0 + std::exp(-g));
        const double h = (double)weight[i] * ((double)x[i] * rinv);
        out[i] = (float)(h * silu_g);
    }
}

void nnml_qwen35_partial_rotary(const float * q_in, float pos, int head_dim,
                                int rotary_dim, float theta, float * q_out) {
    // Degenerate inputs: safe no-op.
    if (q_in == nullptr || q_out == nullptr || head_dim <= 0) {
        return;
    }
    // Clamp rotary_dim into (0, head_dim] and require it even. An odd or
    // non-positive rotary_dim degenerates to a plain copy (no rotation), which
    // is the faithful behavior for rotary_dim=0.
    if (rotary_dim <= 0) {
        for (int i = 0; i < head_dim; ++i) { q_out[i] = q_in[i]; }
        return;
    }
    if (rotary_dim > head_dim) { rotary_dim = head_dim; }
    if ((rotary_dim & 1) != 0) { rotary_dim -= 1; }   // force even
    const int half = rotary_dim / 2;

    // Pass-through dims [rotary_dim .. head_dim) are copied unchanged. Done
    // first so q_out[i] is stable even when q_in == q_out (in-place).
    for (int i = rotary_dim; i < head_dim; ++i) { q_out[i] = q_in[i]; }

    // GPT-NeoX-style rotation over [0, rotary_dim). cos/sin are computed in
    // double for stability; the per-pair math matches HF's
    //   q_embed = q_rot * cos + rotate_half(q_rot) * sin
    //   (emb = cat(freqs, freqs)  =>  cos[i] == cos[i-h], sin[i] == sin[i-h])
    // so pairs (i, i+half) share the same angle:
    //   out[i]      = q[i]*cos - q[i+half]*sin
    //   out[i+half] = q[i+half]*cos + q[i]*sin
    // We read both sources into locals before writing, so in-place (q_in ==
    // q_out) is safe.
    const double dtheta = (double)theta;
    for (int j = 0; j < half; ++j) {
        const double inv_freq =
            1.0 / std::pow(dtheta, (double)(2 * j) / (double)rotary_dim);
        const double angle = (double)pos * inv_freq;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double a = (double)q_in[j];
        const double b = (double)q_in[j + half];
        q_out[j]       = (float)(a * c - b * s);
        q_out[j + half] = (float)(b * c + a * s);
    }
}
