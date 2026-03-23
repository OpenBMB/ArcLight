// definition of minicpm5
#include "models.h"

std::map<std::string, llm_hparams_item> minicpm5_hparams_map = {
    {"general.architecture",                        LLM_ARCHITECTURE},
    {"general.name",                                LLM_NAME},
    {"minicpm5.block_count",                        LLM_N_LAYER},
    {"minicpm5.context_length",                     LLM_CTX_LENGTH},
    {"minicpm5.embedding_length",                   LLM_EMB_LENGTH},
    {"minicpm5.feed_forward_length",                LLM_FFN_LENGTH},
    {"minicpm5.attention.head_count",               LLM_ATTN_HEADCOUNT},
    {"minicpm5.attention.head_count_kv",            LLM_ATTN_HEADCOUNT_KV},
    {"minicpm5.rope.freq_base",                     LLM_ROPE_FREQ_BASE},
    {"minicpm5.attention.layer_norm_rms_epsilon",   LLM_ATTN_LNORM_EPS},
    {"minicpm5.attention.key_length",               LLM_ATTN_KEY_LENGTH},
    {"minicpm5.attention.value_length",             LLM_ATTN_VALUE_LENGTH},
    {"tokenizer.ggml.model",                        LLM_TOKENIZER_MODEL},           // tokenizer properties is defined in tokenizer.h
    {"tokenizer.ggml.pre",                          LLM_TOKENIZER_PRE_MODEL},
    {"tokenizer.ggml.tokens",                       LLM_TOKENIZER_TOKENS},
    {"tokenizer.ggml.scores",                       LLM_TOKENIZER_SCORES},
    {"tokenizer.ggml.token_type",                   LLM_TOKENIZER_TOKEN_TYPE},
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
REGISTER_HPARAMS_MAP(minicpm5, minicpm5_hparams_map);

std::map<std::string, llm_weight_item> minicpm5_weights_map = {
    {"token_embd.weight",        LLM_TOKEN_EMBEDDINGS},
    {"output.weight",            LLM_OUTPUT_PROJ},
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
REGISTER_WEIGHTS_MAP(minicpm5, minicpm5_weights_map);

REGISTER_ROPE_TYPE(minicpm5, llm_rope_type::LLM_ROPE_TYPE_NEOX);
REGISTER_FREQ_BASE(minicpm5, 10000.0f);
REGISTER_FREQ_SCALE(minicpm5, 1.0f);
REGISTER_EXT_FACTOR(minicpm5, 0.0f);
REGISTER_ATTN_FACTOR(minicpm5, 1.0f);
REGISTER_BETA_FAST(minicpm5, 32.0f);
REGISTER_BETA_SLOW(minicpm5, 1.0f);

std::string minicpm5_apply_template(
    const std::vector<chat_msg> & messages,
    bool add_generation_prompt,
    bool enable_reasoning)
{
    // return messages[0].content;
    std::string out;
    out.reserve(messages.size() * 128 + 512);
    // out.append("<|im_start|>system\n");
    // out.append("You are a helpful assistant<|im_end|>\n");

    // first pass: emit all messages but last assistant turn
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];

        if (msg.role == "assistant") {
            // if assistant msg has reasoning content, directly expand it in the history
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
    // generate prompt
    if (add_generation_prompt) {
        out.append("<|im_start|>assistant\n");
        if (enable_reasoning) {
            out.append("<think>\n\n</think>\n\n");
        }
    }
    return out;
}
REGISTER_TEMPLATE_CALLER(minicpm5, minicpm5_apply_template);

void build_minicpm5_forward(nnml_cgraph & graph, llm_model & model, bool is_tp) {
    if (is_tp) assert(graph.get_n_head_kv() % graph.get_n_para_graphs_max() == 0);
    int kv_parallel_size = is_tp ? graph.get_kvcache()->get_n_para_kvcaches() : 1;

    llm_hparams & hparams = graph.get_hparams();
    const int64_t n_embd_head = hparams.n_embd_head_v;
    NNML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    NNML_ASSERT(n_embd_head == hparams.n_rot);

    nnml_tensor * cur;
    nnml_tensor * inpL;
    inpL = graph.build_inp_embd(model.tok_embd);
    nnml_tensor * inp_pos = graph.build_inp_pos(0);
    graph.build_attn_inp_kv();
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    nnml_tensor * inp_out_ids = graph.build_inp_out_ids(0);

    for (int il = 0; il < hparams.n_layer; ++il) {
        nnml_tensor * inpSA = inpL;
        // norm
        cur = graph.build_norm(inpL, model.layers[il].tensors[ATTN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "attn_norm", il);
        // self-attention
        {
            // compute Q and K and RoPE them
            nnml_tensor_ptrs curs;
            if (is_tp) {
                curs = graph.build_scatter(cur, true);
            } else {
                curs = cur;
            }

            nnml_tensor * rope_factors = model.get_rope_factors(il);
            nnml_tensor_ptrs Qcur = graph.build_lora_mm(model.layers[il].tensors[WQ], curs);
            Qcur.set_name("%s-%d", "Qcur", il);
            nnml_tensor_ptrs Kcur = graph.build_lora_mm(model.layers[il].tensors[WK], curs);
            Kcur.set_name("%s-%d", "Kcur", il);
            nnml_tensor_ptrs Vcur = graph.build_lora_mm(model.layers[il].tensors[WV], curs);
            Vcur.set_name("%s-%d", "Vcur", il);
            Qcur = graph.build_reshape_3d(Qcur, n_embd_head, graph.get_n_head()/kv_parallel_size, graph.get_n_tokens());
            Kcur = graph.build_reshape_3d(Kcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());
            Vcur = graph.build_reshape_3d(Vcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());
            
            Qcur = graph.build_rope_ext(Qcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            Qcur.set_name("%s-%d", "Qcur", il);
            Kcur = graph.build_rope_ext(Kcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            Kcur.set_name("%s-%d", "Kcur", il);

            curs = graph.build_attn(model.layers[il].tensors[WO], nullptr, Qcur, Kcur, Vcur, 
                    nullptr, nullptr, nullptr, kq_scale, il);
            curs.set_name("%s-%d", "attn_out", il);
            
            if (is_tp) {
                cur = graph.build_gather(curs, true);
            } else {
                cur = curs;
            }
        }
        if (il == hparams.n_layer - 1 && inp_out_ids && !graph.is_eval_mode()) {
            cur   = graph.build_get_rows(cur, inp_out_ids);
            inpSA = graph.build_get_rows(inpSA, inp_out_ids);
        }
        nnml_tensor * ffn_inp = graph.build_add(cur, inpSA);
        ffn_inp->set_name("%s-%d", "ffn_inp", il);

        // feed-forward network
        cur = graph.build_norm(ffn_inp, model.layers[il].tensors[FFN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "ffn_norm", il);
        
        nnml_tensor_ptrs curs;
        if (is_tp) {
            curs = graph.build_scatter(cur, false);
        } else {
            curs = cur;
        }

        curs = graph.build_ffn(curs,
                        model.layers[il].tensors[FFN_UP],   NULL, NULL,
                        model.layers[il].tensors[FFN_GATE], NULL, NULL,
                        model.layers[il].tensors[FFN_DOWN], NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        curs.set_name("%s-%d", "ffn_out", il);
        
        if (is_tp) {
            cur = graph.build_gather(curs, false);
        } else {
            cur = curs;
        }

        cur = graph.build_add(cur, ffn_inp);
        inpL = cur;
    }
    cur = inpL;
    cur = graph.build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur->set_name("%s-%d", "result_norm", -1);
    // graph.set_t_embd(cur);
    // lm_head
    if (model.output == nullptr) model.output = model.tok_embd;
    cur = graph.build_lora_mm(model.output, cur);
    cur->set_name("%s-%d", "result_output", -1);
    graph.set_t_logits(cur);
}

static bool _reg_minicpm5_builder = (nnml_cgraph::reg_builder("minicpm5", build_minicpm5_forward), true);
