// definition of qwen35 (Qwen3.5 hybrid Gated-DeltaNet linear-attention)
//
// Phase 1 / Task 1: load-only registration.
//   - registers the `qwen35` architecture (hparams map, weights map, rope config,
//     ChatML template, forward-builder name).
//   - the forward builder is a STUB that aborts with "not yet implemented"; the
//     real hybrid builder (full-attn + gated DeltaNet linear-attn dispatch) lands
//     in Task 6. Registering the builder name is what makes ArcLight recognize
//     the architecture at graph-build time.
#include "models.h"

std::map<std::string, llm_hparams_item> qwen35_hparams_map = {
    {"general.architecture",                            LLM_ARCHITECTURE},
    {"general.name",                                    LLM_NAME},
    {"qwen35.block_count",                              LLM_N_LAYER},
    {"qwen35.context_length",                           LLM_CTX_LENGTH},
    {"qwen35.embedding_length",                         LLM_EMB_LENGTH},
    {"qwen35.feed_forward_length",                      LLM_FFN_LENGTH},
    {"qwen35.attention.head_count",                     LLM_ATTN_HEADCOUNT},
    {"qwen35.attention.head_count_kv",                  LLM_ATTN_HEADCOUNT_KV},
    {"qwen35.rope.freq_base",                           LLM_ROPE_FREQ_BASE},
    {"qwen35.attention.layer_norm_rms_epsilon",         LLM_ATTN_LNORM_EPS},
    {"qwen35.attention.key_length",                     LLM_ATTN_KEY_LENGTH},
    {"qwen35.attention.value_length",                   LLM_ATTN_VALUE_LENGTH},
    // partial-rotary mRoPE
    {"qwen35.rope.dimension_count",                     LLM_ROPE_DIMENSION_COUNT},
    {"qwen35.rope.dimension_sections",                  LLM_ROPE_DIMENSION_SECTIONS},
    // ssm / gated DeltaNet linear-attention
    {"qwen35.ssm.conv_kernel",                          LLM_SSM_CONV_KERNEL},
    {"qwen35.ssm.state_size",                           LLM_SSM_STATE_SIZE},
    {"qwen35.ssm.group_count",                          LLM_SSM_GROUP_COUNT},
    {"qwen35.ssm.time_step_rank",                       LLM_SSM_TIME_STEP_RANK},
    {"qwen35.ssm.inner_size",                           LLM_SSM_INNER_SIZE},
    // hybrid layer cadence (full-attention every Nth layer)
    {"qwen35.full_attention_interval",                  LLM_FULL_ATTENTION_INTERVAL},
    {"tokenizer.ggml.model",                            LLM_TOKENIZER_MODEL},
    {"tokenizer.ggml.pre",                              LLM_TOKENIZER_PRE_MODEL},
    {"tokenizer.ggml.tokens",                           LLM_TOKENIZER_TOKENS},
    {"tokenizer.ggml.token_type",                       LLM_TOKENIZER_TOKEN_TYPE},
    {"tokenizer.ggml.merges",                           LLM_TOKENIZER_MERGES},
    {"tokenizer.ggml.eos_token_id",                     LLM_TOKENIZER_EOS_ID},
    {"tokenizer.ggml.padding_token_id",                 LLM_TOKENIZER_PAD_ID},
    {"tokenizer.ggml.bos_token_id",                     LLM_TOKENIZER_BOS_ID},
    {"tokenizer.ggml.add_bos_token",                    LLM_TOKENIZER_ADD_BOS},
    {"tokenizer.chat_template",                         LLM_TOKENIZER_CHAT_TMPL},
    {"general.quantization_version",                    LLM_QUANT_VERSION},
};
REGISTER_HPARAMS_MAP(qwen35, qwen35_hparams_map);

// Weight-name -> item map. ArcLight matches by substring (longest match wins,
// see load_all_tensors), so each key must be a unique substring of the tensor
// name. Note `ssm_a` has no `.weight` suffix in the GGUF; `ssm_dt` is `.bias`.
// Tensors only present on a subset of layers (ssm_*/attn_qkv on linear layers,
// attn_q/k/v/output/q_norm/k_norm on full-attn layers) are mapped all the same
// ArcLight loads whichever are present per layer.
std::map<std::string, llm_weight_item> qwen35_weights_map = {
    {"token_embd.weight",           LLM_TOKEN_EMBEDDINGS},
    {"output_norm.weight",          LLM_OUTPUT_NORM},
    // shared by all 24 layers
    {"attn_norm.weight",            LLM_ATTENTION_NORM},
    {"post_attention_norm.weight",  LLM_POST_ATTENTION_NORM},
    {"ffn_gate.weight",             LLM_FFN_GATE},
    {"ffn_up.weight",               LLM_FFN_UP},
    {"ffn_down.weight",             LLM_FFN_DOWN},
    // full-attention layers (il % 4 == 3): 6 layers
    {"attn_q.weight",               LLM_ATTENTION_Q},
    {"attn_k.weight",               LLM_ATTENTION_K},
    {"attn_v.weight",               LLM_ATTENTION_V},
    {"attn_output.weight",          LLM_ATTENTION_O},
    {"attn_q_norm.weight",          LLM_ATTENTION_Q_NORM},
    {"attn_k_norm.weight",          LLM_ATTENTION_K_NORM},
    // linear-attention layers (il % 4 != 3): 18 layers
    {"attn_qkv.weight",             LLM_ATTENTION_QKV},
    {"ssm_conv1d.weight",           LLM_SSM_CONV1D},
    {"ssm_a",                       LLM_SSM_A},
    {"ssm_alpha.weight",            LLM_SSM_ALPHA},
    {"ssm_beta.weight",             LLM_SSM_BETA},
    {"ssm_dt.bias",                 LLM_SSM_DT},
    {"ssm_norm.weight",             LLM_SSM_NORM},
    {"ssm_out.weight",              LLM_SSM_OUT},
    {"attn_gate.weight",            LLM_ATTENTION_GATE},
};
REGISTER_WEIGHTS_MAP(qwen35, qwen35_weights_map);

// Qwen3.5 full-attention layers use an interleaved mRoPE with section split
// [11,11,10,0] and theta 1e7. ArcLight's LLM_ROPE_TYPE_MROPE implies a 4D
// multimodal position input (n_pos_per_embd()==4), which is not what we want
// for text-only decode in Phase 1. Register NEOX (the closest non-multimodal
// type, matching qwen3) so position allocation stays 1-D; the exact mRoPE
// wiring (partial rotary over the first 64 dims + section split) is Task 4.
REGISTER_ROPE_TYPE(qwen35, llm_rope_type::LLM_ROPE_TYPE_NEOX);
REGISTER_FREQ_BASE(qwen35, 10000000.0f);
REGISTER_FREQ_SCALE(qwen35, 1.0f);
REGISTER_EXT_FACTOR(qwen35, 0.0f);
REGISTER_ATTN_FACTOR(qwen35, 1.0f);
REGISTER_BETA_FAST(qwen35, 32.0f);
REGISTER_BETA_SLOW(qwen35, 1.0f);

// ChatML template (the GGUF tokenizer.chat_template is ChatML with <|im_start|>).
// Reuses the qwen3-style template verbatim.
std::string qwen35_apply_template(
    const std::vector<chat_msg> & messages,
    bool add_generation_prompt,
    bool enable_reasoning)
{
    std::string out;
    out.reserve(messages.size() * 128 + 512);

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];

        if (msg.role == "assistant") {
            if (!msg.reasoning_content.empty()) {
                out.append("<|im_start|>assistant\n");
                out.append("<think>\n");
                out.append(msg.reasoning_content);
                out.append("\n</think>\n\n");
                out.append(msg.content);
                out.append("<|im_end|>\n");
            } else {
                append_block(out, "assistant", msg.content);
            }
        } else if (msg.role == "user") {
            append_block(out, "user", msg.content);
        } else if (msg.role == "system") {
            append_block(out, "system", msg.content);
        }
    }
    if (add_generation_prompt) {
        out.append("<|im_start|>assistant\n");
        if (enable_reasoning) {
            out.append("<think>\n\n</think>\n\n");
        }
    }
    return out;
}
REGISTER_TEMPLATE_CALLER(qwen35, qwen35_apply_template);

// Hybrid forward builder (Task 6).
//   Per-layer dispatch via hparams.is_full_attention(il):
//     - full-attention layers (il % full_attention_interval == 3): GQA + QK-norm
//         + partial-rotary (NEOX, first n_rotary_dim dims) + sigmoid output gate.
//         attn_q is fused [Q | gate] per head (out = 2*n_head*n_embd_head), so Q
//         and the gate are recovered via strided views + cont().
//     - linear-attention layers (the rest): Gated DeltaNet recurrent path -
//         in_proj_qkv -> conv1d state update -> q/k/v split -> delta-rule recurrent
//         update -> gated RMSNorm (norm * silu(in_proj_z)) -> ssm_out projection.
//   Both layer tails share: residual -> post_attention_norm -> SwiGLU FFN -> residual.
//
//   Goal of Task 6: the graph BUILDS and a decode RUNS end-to-end without abort.
//   The SSM conv/delta graph-ops are decode-only (they read/write the token-0 slice
//   of their inputs); prefill tokens > 0 are left untouched, which is sufficient for
//   graph build + single-token decode. Numerical correctness is Task 7.
void build_qwen35_forward(nnml_cgraph & graph, llm_model & model, bool is_tp) {
    llm_hparams & hparams = graph.get_hparams();
    nnml_memory_t & mem = graph.get_mem();
    const int64_t n_tokens    = graph.get_n_tokens();
    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_embd_head = hparams.n_embd_head_v;          // 256 (full-attn head dim)
    const int64_t n_head      = graph.get_n_head();             // 8  (full-attn)
    const int64_t n_head_kv   = graph.get_n_head_kv();          // 2  (full-attn, GQA)
    // partial rotary: rotate only the first n_rotary_dim head dims (64); fall back to n_rot.
    const int   n_rot     = hparams.n_rotary_dim > 0 ? (int)hparams.n_rotary_dim : (int)graph.get_n_rot();
    const float kq_scale  = 1.0f / sqrtf(float(n_embd_head));

    NNML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    NNML_ASSERT(model.ssm_state != nullptr);

    // emit the derived hybrid layer-type pattern so load verification can confirm
    // the L,L,L,F cadence without a layer_types array in the GGUF.
    uint32_t n_full = 0, n_linear = 0;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.is_full_attention(il)) ++n_full; else ++n_linear;
    }
    NNML_ASSERT(n_linear == model.ssm_state->conv_states.size());
    NNML_ASSERT(n_linear == model.ssm_state->recurrent_states.size());

    nnml_tensor * cur;
    nnml_tensor * inpL = graph.build_inp_embd(model.tok_embd);
    nnml_tensor * inp_pos = graph.build_inp_pos(0);
    graph.build_attn_inp_kv();
    nnml_tensor * inp_out_ids = graph.build_inp_out_ids(0);

    uint32_t linear_idx = 0;

    for (int il = 0; il < hparams.n_layer; ++il) {
        nnml_tensor * inpSA = inpL;
        cur = graph.build_norm(inpL, model.layers[il].tensors[ATTN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "attn_norm", il);

        if (hparams.is_full_attention(il)) {
            // ===== FULL ATTENTION: GQA + QK-norm + partial-rotary + output gate =====
            // attn_q is fused [Q | gate] per head => out_features = 2 * n_head * n_embd_head.
            nnml_tensor * qg = graph.build_lora_mm(model.layers[il].tensors[WQ][0], cur);  // [2*n_head*n_embd_head, n_tok]
            qg->set_name("%s-%d", "qg", il);
            const size_t esz          = qg->element_size();
            const int64_t stride_head = 2 * n_embd_head;              // 512  (q + gate per head)
            const int64_t stride_tok  = 2 * n_head * n_embd_head;     // 4096 (full fused block per token)
            // Fused layout per token: [head0_Q(256), head0_gate(256), head1_Q(256), head1_gate(256), ...].
            // view_3d(a, ne0, ne1, ne2, nb1, nb2, offset) with nb0 implicit (esz): nb1 is the
            // axis-1 (heads) byte stride = stride_head*esz (512 between heads), nb2 is the
            // axis-2 (tokens) byte stride = stride_tok*esz (4096 between tokens). The gate
            // view reuses the same strides with offset n_embd_head*esz (the gate half).
            nnml_tensor * Qv = nnml_view_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, qg,
                    n_embd_head, n_head, n_tokens, stride_head * esz, stride_tok * esz, 0);
            nnml_tensor * Qcur = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, Qv);
            graph.build_forward_expand(Qcur);
            nnml_tensor * Gv = nnml_view_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, qg,
                    n_embd_head, n_head, n_tokens, stride_head * esz, stride_tok * esz, n_embd_head * esz);
            nnml_tensor * gate = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, Gv);
            graph.build_forward_expand(gate);   // the cont is a raw op: it must be expanded or it stays zero
            gate = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, gate, n_embd, n_tokens);
            graph.build_forward_expand(gate);

            // QK-norm (RMSNorm over head_dim).
            Qcur = graph.build_norm(Qcur, model.layers[il].tensors[ATTN_Q_NORM][0], NULL, LLM_NORM_RMS, il);
            // K, V (GQA: n_head_kv heads each).
            nnml_tensor * Kcur = graph.build_lora_mm(model.layers[il].tensors[WK][0], cur);
            Kcur = graph.build_reshape_3d(Kcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = graph.build_norm(Kcur, model.layers[il].tensors[ATTN_K_NORM][0], NULL, LLM_NORM_RMS, il);
            nnml_tensor * Vcur = graph.build_lora_mm(model.layers[il].tensors[WV][0], cur);
            Vcur = graph.build_reshape_3d(Vcur, n_embd_head, n_head_kv, n_tokens);
            // partial rotary (NEOX, first n_rotary_dim=64 of each head's 256 dims).
            Qcur = graph.build_rope_ext(Qcur, inp_pos, nullptr, hparams.n_rotary_dim, graph.get_rope_type(),
                    graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            Kcur = graph.build_rope_ext(Kcur, inp_pos, nullptr, hparams.n_rotary_dim, graph.get_rope_type(),
                    graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            // GQA attention, then gate BEFORE the output projection. Reference order:
            //   out = WO( softmax(QK^T·scale)·V ⊙ sigmoid(gate) ).
            // build_attn with wo=nullptr returns the raw attention output [n_embd, n_tok];
            // gate it, then apply WO.
            cur = graph.build_attn(nullptr, nullptr, Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr, kq_scale, il);
            nnml_tensor * gate_s = nnml_unary(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, gate, NNML_UNARY_OP_SIGMOID);
            graph.build_forward_expand(gate_s);
            cur = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, gate_s);
            graph.build_forward_expand(cur);
            cur = graph.build_lora_mm(model.layers[il].tensors[WO][0], cur);
        } else {
            // ===== LINEAR ATTENTION: Gated DeltaNet =====
            NNML_ASSERT(linear_idx < model.ssm_state->conv_states.size());
            nnml_tensor * conv_state = model.ssm_state->conv_states[linear_idx];
            nnml_tensor * rec_state  = model.ssm_state->recurrent_states[linear_idx];
            const int64_t v_dim      = rec_state->get_elements(0);   // 128 (head_v_dim)
            const int64_t k_dim      = rec_state->get_elements(1);   // 128 (head_k_dim)
            const int64_t ssm_heads  = rec_state->get_elements(2);   // 16  (num_heads)
            const int64_t key_dim    = ssm_heads * k_dim;            // 2048

            // in_proj_qkv -> depthwise conv1d recurrent state update.
            nnml_tensor * qkv = graph.build_lora_mm(model.layers[il].tensors[ATTENTION_QKV][0], cur);  // [conv_dim, n_tok]
            qkv->set_name("%s-%d", "ssm_qkv", il);
            const int64_t conv_dim = qkv->get_elements(0);           // 2*key_dim + value_dim = 6144
            nnml_tensor * conv_out = nnml_ssm_conv_update(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0,
                    qkv, conv_state, model.layers[il].tensors[SSM_CONV1D][0]);
            graph.build_forward_expand(conv_out);
            conv_out->set_name("%s-%d", "ssm_conv_out", il);
            const size_t esz = conv_out->element_size();

            // split q/k/v out of conv_out via strided views (compute reads token-0 flat slice).
            // Split q/k/v out of conv_out. conv_out is [conv_dim, n_tokens] where
            // conv_dim = 2*key_dim + value_dim, and within each segment the layout is
            // [head0(k_dim), head1(k_dim), ...] head-major. view_3d(a,ne0,ne1,ne2,nb1,nb2,off)
            // with nb0 implicit (esz): nb1 = per-head byte stride (k_dim/v_dim elements),
            // nb2 = per-token byte stride (conv_dim elements). Byte offset selects q (0),
            // k (key_dim), or v (2*key_dim).
            nnml_tensor * q = nnml_view_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, conv_out,
                    k_dim, ssm_heads, n_tokens, k_dim * esz, conv_dim * esz, 0);
            nnml_tensor * k = nnml_view_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, conv_out,
                    k_dim, ssm_heads, n_tokens, k_dim * esz, conv_dim * esz, key_dim * esz);
            nnml_tensor * v = nnml_view_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, conv_out,
                    v_dim, ssm_heads, n_tokens, v_dim * esz, conv_dim * esz, 2 * key_dim * esz);

            // g = -exp(A_log) * softplus(alpha + dt_bias).  softplus(x) = log(1 + exp(x)).
            // NOTE: ssm_a (LLM_SSM_A) is the NOSCAN form and already stores -exp(A_log)
            // (verified: all per-head values are negative). The delta-rule op applies
            // exp(g) internally as the state decay, so g is ssm_a * softplus(...) directly
            // — do NOT exp()/negate ssm_a (that would double-transform the decay).
            nnml_tensor * a       = graph.build_lora_mm(model.layers[il].tensors[SSM_ALPHA][0], cur);  // [ssm_heads, n_tok]
            nnml_tensor * b       = graph.build_lora_mm(model.layers[il].tensors[SSM_BETA ][0], cur);  // [ssm_heads, n_tok]
            nnml_tensor * ssm_a   = model.layers[il].tensors[SSM_A ][0];   // [ssm_heads], holds -exp(A_log)
            nnml_tensor * dt_bias = model.layers[il].tensors[SSM_DT][0];   // [ssm_heads]
            nnml_tensor * a_dt    = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, a, dt_bias);      // broadcast dt_bias over tokens
            graph.build_forward_expand(a_dt);
            nnml_tensor * ex      = nnml_unary(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, a_dt,    NNML_UNARY_OP_EXP);
            graph.build_forward_expand(ex);
            nnml_tensor * one_pex = nnml_scale_bias(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, ex, 1.0f, 1.0f);  // 1 + exp(x)
            graph.build_forward_expand(one_pex);
            nnml_tensor * sp      = nnml_log(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, one_pex);                // softplus
            graph.build_forward_expand(sp);
            // g = ssm_a * softplus(...); src1 = ssm_a (per-head) broadcasts over tokens.
            nnml_tensor * g       = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, sp, ssm_a);
            graph.build_forward_expand(g);
            nnml_tensor * beta    = nnml_unary(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, b, NNML_UNARY_OP_SIGMOID);
            graph.build_forward_expand(beta);

            // delta-rule recurrent state update -> [v_dim, ssm_heads, n_tok].
            nnml_tensor * delta_out = nnml_ssm_delta_update(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0,
                    q, k, v, g, beta, rec_state);
            graph.build_forward_expand(delta_out);
            delta_out->set_name("%s-%d", "ssm_delta_out", il);

            // gated RMSNorm: out = RMSNorm(delta_out, ssm_norm) * silu(in_proj_z).
            nnml_tensor * normed = graph.build_norm(delta_out, model.layers[il].tensors[SSM_NORM][0], NULL, LLM_NORM_RMS, il);
            nnml_tensor * z_mm   = graph.build_lora_mm(model.layers[il].tensors[ATTENTION_GATE][0], cur);   // [value_dim, n_tok]
            nnml_tensor * z      = graph.build_reshape_3d(z_mm, v_dim, ssm_heads, n_tokens);          // [v_dim, ssm_heads, n_tok]
            nnml_tensor * silu_z = nnml_unary(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, z, NNML_UNARY_OP_SILU);
            graph.build_forward_expand(silu_z);
            nnml_tensor * gated  = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, normed, silu_z);
            graph.build_forward_expand(gated);
            // ssm_out projection: reshape [v_dim, ssm_heads, n_tok] -> [value_dim, n_tok] -> [n_embd, n_tok].
            nnml_tensor * gated_2d = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0,
                    gated, ssm_heads * v_dim, n_tokens);
            graph.build_forward_expand(gated_2d);
            cur = graph.build_lora_mm(model.layers[il].tensors[SSM_OUT][0], gated_2d);
            cur->set_name("%s-%d", "ssm_out", il);
            ++linear_idx;
        }

        // common tail: residual -> post_attention_norm -> SwiGLU FFN -> residual.
        if (il == hparams.n_layer - 1 && inp_out_ids && !graph.is_eval_mode()) {
            cur   = graph.build_get_rows(cur, inp_out_ids);
            inpSA = graph.build_get_rows(inpSA, inp_out_ids);
        }
        nnml_tensor * ffn_inp = graph.build_add(cur, inpSA);
        ffn_inp->set_name("%s-%d", "ffn_inp", il);
        cur = graph.build_norm(ffn_inp, model.layers[il].tensors[POST_ATTENTION_NORM][0], NULL, LLM_NORM_RMS, il);
        cur = graph.build_ffn(cur,
                model.layers[il].tensors[FFN_UP][0],   NULL, NULL,
                model.layers[il].tensors[FFN_GATE][0], NULL, NULL,
                model.layers[il].tensors[FFN_DOWN][0], NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = graph.build_add(cur, ffn_inp);
        inpL = cur;
    }

    cur = graph.build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur->set_name("%s-%d", "result_norm", -1);
    // lm_head (tied embeddings).
    if (model.output == nullptr) model.output = model.tok_embd;
    cur = graph.build_lora_mm(model.output, cur);
    cur->set_name("%s-%d", "result_output", -1);
    graph.set_t_logits(cur);

    (void)is_tp;
}

static bool _reg_qwen35_builder = (nnml_cgraph::reg_builder("qwen35", build_qwen35_forward), true);
