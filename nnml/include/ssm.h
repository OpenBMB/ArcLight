#pragma once

#include <cstddef>   // size_t

// Stateful short-convolution primitives for linear-attention / SSM layers.
//
// These are standalone, portable-scalar compute functions (no graph node, no
// backend dispatch) mirroring nnml/src/ops/arm/tq2_i2s.{cpp,h}: small, focused,
// unit-testable building blocks that a later task will wire into the graph /
// stateful decode path.

// Depthwise causal Conv1d update for a single token (decode step).
//
// Matches HuggingFace's `torch_causal_conv1d_update` for the Qwen3.5
// GatedDeltaNet layers (transformers/models/qwen3_5/modeling_qwen3_5.py):
//   hidden_states_new = cat([conv_state, x], dim=-1)        # length kernel+1
//   conv_state  = hidden_states_new[:, :, -kernel:]         # shift left 1, append x
//   out         = SiLU(conv1d(hidden_states_new, weight, groups=conv_dim)[:, :, -1:])
//
// For the decode case (one new input x of length conv_dim), this collapses to:
//   1. Roll conv_state left by one tap (drop index 0), store x at the last tap.
//   2. out[c] = SiLU( sum_{k=0..kernel-1} conv_state[c, k] * weight[c, k] ).
//
// Memory layout (row-major for all 2-D buffers):
//   conv_state : [conv_dim * kernel] f32, the recurrent state holding the last
//                `kernel` inputs per channel. conv_state[c * kernel + k].
//                UPDATED IN PLACE so the next token sees the rolled state.
//   weight     : [conv_dim * kernel] f32, depthwise Conv1d weights in
//                [conv_dim, kernel] (out_channels, kernel) layout — i.e. the
//                same layout HF indexes after `.squeeze(1)`. If a GGUF stores
//                these transposed ([kernel, conv_dim]), the loader must transpose
//                into this layout before calling; the op assumes [conv_dim, kernel].
//
// Args:
//   x          : [conv_dim] f32, this token's (pre-conv) qkv projection.
//   conv_state : [conv_dim * kernel] f32, recurrent state (see above). MUTATED.
//   weight     : [conv_dim * kernel] f32, depthwise conv weights (see above).
//   conv_dim   : number of channels (= 2*key_dim + value_dim per layer).
//   kernel     : conv kernel size (= ssm.conv_kernel = 4 for Qwen3.5).
//   out        : [conv_dim] f32 output (WRITTEN).
void nnml_ssm_conv1d_update(const float * x, float * conv_state, const float * weight,
                            int conv_dim, int kernel, float * out);

// Multi-token depthwise causal Conv1d update (prefill path).
//
// Processes `n_tokens` inputs for channels [ith, ith+nth, ...) — i.e. the channel
// axis is parallelized across threads, the token axis is sequential (the conv_state
// rolls left by one tap per token, matching the single-token primitive). This is
// the parallel-over-channels form of `nnml_ssm_conv1d_update`.
//
// Layouts (row-major): x is [conv_dim, n_tokens] (conv_dim contiguous, token stride
// x_nb1 bytes); out is [conv_dim, n_tokens] (out_nb1 bytes). conv_state / weight are
// [conv_dim, kernel] as in the single-token primitive; conv_state is MUTATED to hold
// the last `kernel` inputs after the final token.
void nnml_ssm_conv1d_update_seq(const float * x, size_t x_nb1,
                                float * conv_state, const float * weight,
                                int conv_dim, int kernel, int n_tokens,
                                float * out, size_t out_nb1, int ith, int nth);

// Gated delta-rule recurrent update for ONE head, ONE token (decode step).
//
// Matches HuggingFace's `torch_recurrent_gated_delta_rule` for the Qwen3.5
// GatedDeltaNet linear-attention layers
// (transformers/models/qwen3_5/modeling_qwen3_5.py:404-443), collapsed to the
// single-token / per-head case. Per head h, per token, with recurrent state
// S_h in R^{k_dim x v_dim}:
//
//   1. q_n = l2norm(q) * (1/sqrt(k_dim))     # l2norm + query scale
//      k_n = l2norm(k)
//      where l2norm(x) = x * rsqrt(sum(x^2) + 1e-6)   (HF eps = 1e-6)
//   2. S <- exp(g) * S                        # gated decay (g is the raw scalar)
//   3. kv_mem[v] = sum_k S[k,v] * k_n[k]      # contract the k axis -> [v_dim]
//   4. delta[v] = (v[v] - kv_mem[v]) * beta
//   5. S[k,v] += k_n[k] * delta[v]            # rank-1 outer-product update
//   6. out[v] = sum_k S[k,v] * q_n[k]         # contract the k axis -> [v_dim]
//
// Memory layout (row-major):
//   S : [k_dim * v_dim] f32, row-major [k_dim, v_dim] (S[k * v_dim + v]).
//       IN/OUT — mutated in place and persists across tokens (the recurrent
//       state of the delta rule). The decay in step 2 and the rank-1 update in
//       step 5 both write through to S before the output is read in step 6, so
//       `out` is computed against the post-update state (matches HF).
//   q, k : [k_dim] f32 raw inputs (L2-norm + scale applied internally).
//   v    : [v_dim] f32 raw inputs.
//   g, beta : scalars for this head/token (g is pre-exp; beta in [0, 1]).
//
// Args:
//   q, k : [k_dim] f32 raw query / key for this token.
//   v    : [v_dim] f32 raw value for this token.
//   g    : per-head/per-token gate (raw; the op applies exp(g)).
//   beta : per-head/per-token delta-rule beta.
//   S    : [k_dim * v_dim] f32 recurrent state (see above). MUTATED.
//   k_dim, v_dim : head dims (= 128 for Qwen3.5-2B).
//   out  : [v_dim] f32 output (WRITTEN).
void nnml_ssm_gated_delta_rule_update(const float * q, const float * k, const float * v,
                                      float g, float beta, float * S,
                                      int k_dim, int v_dim, float * out);

// Multi-token gated delta-rule recurrence (prefill path).
//
// For heads [ith, ith+nth, ...) runs the AR update over all n_tokens (the token
// axis is sequential — S mutates per token; the head axis is parallel across
// threads). This is the parallel-over-heads form of nnml_ssm_gated_delta_rule_update.
//
// Layouts: q,k are [k_dim, num_heads, n_tokens]; v is [v_dim, num_heads, n_tokens];
// g,beta are [num_heads, n_tokens]; out is [v_dim, num_heads, n_tokens]; S is
// [v_dim, k_dim, num_heads] (one k_dim x v_dim block per head), MUTATED. q_nb2 /
// k_nb2 / v_nb2 / out_nb2 are per-token byte strides (nb[2]); g_nb1 / b_nb1 are
// g/beta per-token byte strides (nb[1]).
void nnml_ssm_delta_rule_update_seq(const float * q, size_t q_nb2, const float * k, size_t k_nb2,
                                    const float * v, size_t v_nb2, const float * g, size_t g_nb1,
                                    const float * beta, size_t b_nb1, float * S,
                                    int k_dim, int v_dim, int num_heads, int n_tokens,
                                    float * out, size_t out_nb2, int ith, int nth);

// Gated RMSNorm for the linear-attention layers' output projection input.
//
// Matches HuggingFace's `Qwen3_5RMSNormGated.forward`
// (transformers/models/qwen3_5/modeling_qwen3_5.py:271-280), computed in
// float32 throughout:
//   variance = mean(x*x)                                 # over the last axis
//   h        = x * rsqrt(variance + eps)                 # plain RMSNorm
//   h        = weight * h                                # elementwise scale
//   out      = h * SiLU(gate)                            # SiLU(g)=g/(1+exp(-g))
// with eps = 1e-6 (Qwen3.5RMSNormGated default).
//
// This is the norm used inside Qwen3.5GatedDeltaNet just before out_proj:
// the gate is the layer's `z` projection (the same shape as `x`).
//
// Args:
//   x      : [n] f32 input hidden states.
//   weight : [n] f32 learned RMSNorm scale (the `norm.weight` tensor).
//   gate   : [n] f32 gate pre-activation (the `z` projection; SiLU applied here).
//   n      : number of elements (== head_v_dim total over heads for the layer).
//   eps    : RMSNorm epsilon (1e-6 for Qwen3.5).
//   out    : [n] f32 output (WRITTEN; may alias x).
void nnml_ssm_rms_norm_gated(const float * x, const float * weight, const float * gate,
                             int n, float eps, float * out);

// Partial-rotary mRoPE for ONE query/key head vector, ONE token (Qwen3.5
// full-attention layers). Standalone, scalar, decode-friendly.
//
// Matches HuggingFace's `Qwen3_5TextRotaryEmbedding.forward` (175-245) +
// `apply_rotary_pos_emb` (630-673) for **text-only** position ids, with:
//   head_dim = 256, partial_rotary_factor = 0.25  -> rotary_dim = 64
//   rope_theta = 1e7, rope_type = default (GPT-NeoX-style `rotate_half`)
//   mrope_section = [11, 11, 10]
//
// Rotary convention (verified against HF, modeling_qwen3_5.py:630-673):
//   inv_freq[j]   = 1.0 / (theta ^ (2j / rotary_dim)),  j in [0, rotary_dim/2)
//   angle[j]      = pos * inv_freq[j]                   (32 distinct angles)
//   cos[i]=cos(angle[i mod h]), sin[i]=sin(angle[i mod h])  where h=rotary_dim/2
//     (HF builds emb = cat(freqs, freqs), so the cos/sin tensor is the same
//      value at i and i+h — see modeling_qwen3_5.py:241.)
//   For i in [0, h):     out[i]   = q[i]   * cos[i] - q[i+h] * sin[i]
//   For i in [h, rot):   out[i]   = q[i]   * cos[i] + q[i-h] * sin[i]
//   For i in [rot, head_dim): out[i] = q[i]   (partial rotary: pass-through)
//
// mRoPE section handling — IMPORTANT ASSUMPTION for text-only:
// HF's `Qwen3_5TextRotaryEmbedding.forward` expands 1-D position_ids across the
// 3 mRoPE axes (T/H/W) — `position_ids[None,...].expand(3, ...)` — so for
// text-only decode all three axes hold the *same* position. HF's
// `apply_interleaved_mrope` then permutes the per-axis freqs into the
// interleaved [THWTHW...TT] layout, but since every axis is identical the
// permutation is a no-op (verified: `torch.allclose(freqs_t, freqs[0])` ==
// True). Therefore the section split does not affect text-only output and this
// helper implements plain partial-rotary GPT-NeoX RoPE. (If a future caller
// ever feeds distinct T/H/W positions — multimodal — this helper must be
// extended to take 3 positions and apply the section.)
//
// Args:
//   q_in       : [head_dim] f32 input query (or key) vector for one head.
//   pos        : scalar token position (text-only; same value on all 3 mRoPE
//                axes per the assumption above).
//   head_dim   : full per-head dim (256 for Qwen3.5-2B).
//   rotary_dim : # leading dims to rotate (64 for Qwen3.5-2B). Must be even
//                and <= head_dim; clamped otherwise.
//   theta      : RoPE base (1e7 for Qwen3.5).
//   q_out      : [head_dim] f32 output (WRITTEN; may alias q_in).
void nnml_qwen35_partial_rotary(const float * q_in, float pos, int head_dim,
                                int rotary_dim, float theta, float * q_out);
