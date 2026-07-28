// definition of minicpm (BitCPM / MiniCPM4 ternary, architecture key "minicpm")
#include "models.h"

std::map<std::string, llm_hparams_item> minicpm_hparams_map = {
    {"general.architecture",                        LLM_ARCHITECTURE},
    {"general.name",                                LLM_NAME},
    {"minicpm.block_count",                         LLM_N_LAYER},
    {"minicpm.context_length",                      LLM_CTX_LENGTH},
    {"minicpm.embedding_length",                    LLM_EMB_LENGTH},
    {"minicpm.feed_forward_length",                 LLM_FFN_LENGTH},
    {"minicpm.attention.head_count",                LLM_ATTN_HEADCOUNT},
    {"minicpm.attention.head_count_kv",             LLM_ATTN_HEADCOUNT_KV},
    {"minicpm.rope.freq_base",                      LLM_ROPE_FREQ_BASE},
    {"minicpm.attention.layer_norm_rms_epsilon",    LLM_ATTN_LNORM_EPS},
    {"minicpm.attention.key_length",                LLM_ATTN_KEY_LENGTH},
    {"minicpm.attention.value_length",              LLM_ATTN_VALUE_LENGTH},
    {"minicpm.embedding_scale",                     LLM_EMBEDDING_SCALE},
    {"minicpm.residual_scale",                      LLM_RESIDUAL_SCALE},
    {"minicpm.logit_scale",                         LLM_LOGIT_SCALE},
    {"tokenizer.ggml.model",                        LLM_TOKENIZER_MODEL},
    {"tokenizer.ggml.pre",                          LLM_TOKENIZER_PRE_MODEL},
    {"tokenizer.ggml.tokens",                       LLM_TOKENIZER_TOKENS},
    {"tokenizer.ggml.scores",                       LLM_TOKENIZER_SCORES},
    {"tokenizer.ggml.token_type",                   LLM_TOKENIZER_TOKEN_TYPE},
    {"tokenizer.ggml.merges",                       LLM_TOKENIZER_MERGES},
    {"tokenizer.ggml.eos_token_id",                 LLM_TOKENIZER_EOS_ID},
    {"tokenizer.ggml.padding_token_id",             LLM_TOKENIZER_PAD_ID},
    {"tokenizer.ggml.unknown_token_id",             LLM_TOKENIZER_UNK_ID},
    {"tokenizer.ggml.bos_token_id",                 LLM_TOKENIZER_BOS_ID},
    {"tokenizer.ggml.add_bos_token",                LLM_TOKENIZER_ADD_BOS},
    {"tokenizer.ggml.add_eos_token",                LLM_TOKENIZER_ADD_EOS},
    {"tokenizer.ggml.add_sep_token",                LLM_TOKENIZER_ADD_SEP},
    {"tokenizer.ggml.add_space_prefix",             LLM_TOKENIZER_ADD_SPA},
    {"tokenizer.chat_template",                     LLM_TOKENIZER_CHAT_TMPL},
    {"general.quantization_version",                LLM_QUANT_VERSION},
};
REGISTER_HPARAMS_MAP(minicpm, minicpm_hparams_map);

std::map<std::string, llm_weight_item> minicpm_weights_map = {
    {"token_embd.weight",        LLM_TOKEN_EMBEDDINGS},
    {"output.weight",            LLM_OUTPUT_PROJ},     // absent when tied (0.5B) — harmless
    {"output_norm.weight",       LLM_OUTPUT_NORM},
    {"attn_norm.weight",         LLM_ATTENTION_NORM},
    {"attn_q.weight",            LLM_ATTENTION_Q},
    {"attn_k.weight",            LLM_ATTENTION_K},
    {"attn_v.weight",            LLM_ATTENTION_V},
    {"attn_output.weight",       LLM_ATTENTION_O},
    {"ffn_down.weight",          LLM_FFN_DOWN},
    {"ffn_up.weight",            LLM_FFN_UP},
    {"ffn_gate.weight",          LLM_FFN_GATE},
    {"ffn_norm.weight",          LLM_FFN_NORM},
    {"rope_factors_long.weight", LLM_ROPE_LONG},
    {"rope_factors_short.weight",LLM_ROPE_SHORT},
};
REGISTER_WEIGHTS_MAP(minicpm, minicpm_weights_map);

REGISTER_ROPE_TYPE(minicpm, llm_rope_type::LLM_ROPE_TYPE_NORM);
REGISTER_FREQ_BASE(minicpm, 10000.0f);
REGISTER_FREQ_SCALE(minicpm, 1.0f);
REGISTER_EXT_FACTOR(minicpm, 0.0f);
REGISTER_ATTN_FACTOR(minicpm, 1.0f);
REGISTER_BETA_FAST(minicpm, 32.0f);
REGISTER_BETA_SLOW(minicpm, 1.0f);

std::string minicpm_apply_template(
    const std::vector<chat_msg> & messages,
    bool add_generation_prompt,
    bool enable_reasoning)
{
    std::string out;
    out.reserve(messages.size() * 128 + 512);
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];
        if (msg.role == "assistant") {
            append_block(out, "assistant", msg.content);
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
REGISTER_TEMPLATE_CALLER(minicpm, minicpm_apply_template);

void build_minicpm_forward(nnml_cgraph & graph, llm_model & model, bool is_tp) {
    if (is_tp) assert(graph.get_n_head_kv() % graph.get_n_para_graphs_max() == 0);
    int kv_parallel_size = is_tp ? graph.get_kvcache()->get_n_para_kvcaches() : 1;

    llm_hparams & hparams = graph.get_hparams();
    const int64_t n_embd_head = hparams.n_embd_head_v;
    NNML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    NNML_ASSERT(n_embd_head == hparams.n_rot);

    // embedding_scale is applied inside build_inp_embd (cgraph.cpp:58) when non-zero.
    nnml_tensor * cur;
    nnml_tensor * inpL = graph.build_inp_embd(model.tok_embd);
    nnml_tensor * inp_pos = graph.build_inp_pos(0);
    graph.build_attn_inp_kv();
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    nnml_tensor * inp_out_ids = graph.build_inp_out_ids(0);

    for (int il = 0; il < hparams.n_layer; ++il) {
        nnml_tensor * inpSA = inpL;
        cur = graph.build_norm(inpL, model.layers[il].tensors[ATTN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "attn_norm", il);
        {
            nnml_tensor_ptrs curs;
            if (is_tp) { curs = graph.build_scatter(cur, true); }
            else       { curs = cur; }

            nnml_tensor * rope_factors = model.get_rope_factors(il);
            nnml_tensor_ptrs Qcur = graph.build_lora_mm(model.layers[il].tensors[WQ], curs);
            nnml_tensor_ptrs Kcur = graph.build_lora_mm(model.layers[il].tensors[WK], curs);
            nnml_tensor_ptrs Vcur = graph.build_lora_mm(model.layers[il].tensors[WV], curs);
            Qcur = graph.build_reshape_3d(Qcur, n_embd_head, graph.get_n_head()/kv_parallel_size, graph.get_n_tokens());
            Kcur = graph.build_reshape_3d(Kcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());
            Vcur = graph.build_reshape_3d(Vcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());

            Qcur = graph.build_rope_ext(Qcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            Kcur = graph.build_rope_ext(Kcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());

            curs = graph.build_attn(model.layers[il].tensors[WO], nullptr, Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr, kq_scale, il);
            if (is_tp) {
                cur = graph.build_gather(curs, true);
            } else {
                cur = curs;
            }
        }
        // MiniCPM depth scale on the attention output (residual branch)
        if (hparams.f_residual_scale != 0.0f) {
            cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_residual_scale);
            graph.build_forward_expand(cur);
        }
        if (il == hparams.n_layer - 1 && inp_out_ids && !graph.is_eval_mode()) {
            cur   = graph.build_get_rows(cur, inp_out_ids);
            inpSA = graph.build_get_rows(inpSA, inp_out_ids);
        }
        nnml_tensor * ffn_inp = graph.build_add(cur, inpSA);
        ffn_inp->set_name("%s-%d", "ffn_inp", il);

        cur = graph.build_norm(ffn_inp, model.layers[il].tensors[FFN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "ffn_norm", il);

        nnml_tensor_ptrs curs;
        if (is_tp) { curs = graph.build_scatter(cur, false); }
        else       { curs = cur; }

        curs = graph.build_ffn(curs,
                        model.layers[il].tensors[FFN_UP],   NULL, NULL,
                        model.layers[il].tensors[FFN_GATE], NULL, NULL,
                        model.layers[il].tensors[FFN_DOWN], NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        if (is_tp) {
            cur = graph.build_gather(curs, false);
        } else {
            cur = curs;
        }

        // MiniCPM depth scale on the FFN output (residual branch)
        if (hparams.f_residual_scale != 0.0f) {
            cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_residual_scale);
            graph.build_forward_expand(cur);
        }

        cur = graph.build_add(cur, ffn_inp);
        inpL = cur;
    }
    cur = inpL;
    cur = graph.build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur->set_name("%s-%d", "result_norm", -1);
    // tied output when output.weight is absent (0.5B)
    if (model.output == nullptr) model.output = model.tok_embd;
    cur = graph.build_lora_mm(model.output, cur);
    cur->set_name("%s-%d", "result_output", -1);
    // MiniCPM output logit scale
    if (hparams.f_logit_scale != 1.0f && hparams.f_logit_scale != 0.0f) {
        cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_logit_scale);
        graph.build_forward_expand(cur);
    }
    graph.set_t_logits(cur);
}

static bool _reg_minicpm_builder = (nnml_cgraph::reg_builder("minicpm", build_minicpm_forward), true);
