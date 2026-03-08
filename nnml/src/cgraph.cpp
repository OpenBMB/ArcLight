/**
 * @file cgraph.cpp
 * @brief The implementation of nnml_cgraph class, llm_kv_cache class.
 * 
 * This file implements the nnml_cgraph class for building the computation graph for LLM inference, 
 * and the llm_kv_cache class which manages the key-value cache for attention mechanism.
 */
#include <algorithm>
#include <time.h>

#include "cgraph.h"


NNML_API int64_t nnml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}


// model register
std::unordered_map<std::string, std::function<void(nnml_cgraph &, void *, bool)>> nnml_cgraph::registry;


// implementation of nnml_cgraph
nnml_cgraph::nnml_cgraph(
        int64_t n_weight_tensors, llm_hparams &  hparams,        nnml_memory_t & mem,
        bool    is_tp,            int            n_sub_graphs,   std::string     name,
        bool    flash_attn,       bool           is_dual_buffer, bool    kv_unified,
        llm_kv_cache * kv_cache,  bool           is_eval) noexcept
    : hparams(hparams), mem(std::move(mem)), is_tp(is_tp), name(name), flash_attn(flash_attn), is_dual_buffer(is_dual_buffer),
      kv_unified(kv_unified), kvcache(kv_cache), is_para_mode(false), n_para_graphs(1), n_para_graphs_max(n_sub_graphs),
      n_tokens(0), n_nodes(0), n_outputs(1), n_kv_max(0), is_eval(is_eval)
{
    if (!is_tp) NNML_ASSERT(n_sub_graphs == 1);
    // std::printf("nnml_cgraph: n_sub_graphs=%d\n", n_sub_graphs);

    n_cnode_array = static_cast<int32_t>(n_weight_tensors * (6 + n_sub_graphs));
    nodes = new nnml_cgraph_cnode[n_cnode_array];
}

nnml_tensor * nnml_cgraph::build_inp_embd(nnml_tensor * tok_embd) {
    n_tokens = ubatch.n_tokens;
    const int64_t n_embd = hparams.get_n_embd();
    nnml_tensor * cur = nullptr;

    if (ubatch.token) {
        t_tokens = tensor_new_1d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, NNML_TYPE_I32, ubatch.n_tokens);
        cur = nnml_get_rows(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, tok_embd, t_tokens);
    } else {
        t_embd = tensor_new_2d(mem, NNML_TENSOR_TYPE_WEIGHT, 0, 0, NNML_TYPE_F32, n_embd, ubatch.n_tokens);
        cur = t_embd;
    }
    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_embedding_scale);
    }

    cur->set_name("%s-%d", "inp_embd", -1);
    build_forward_expand(cur);
    return cur;
}

nnml_tensor * nnml_cgraph::build_inp_pos(int i) {
    nnml_tensor * cur = tensor_new_1d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, NNML_TYPE_I32, (int64_t)n_tokens * hparams.n_pos_per_embd());
    t_pos = cur;
    build_forward_expand(cur);
    return cur;
}

nnml_tensor * nnml_cgraph::build_inp_out_ids(int i) {
    nnml_tensor * cur = tensor_new_1d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, NNML_TYPE_I32, n_outputs);
    t_out_ids = cur;
    return cur;
}

void nnml_cgraph::build_attn_inp_kv_impl(const llm_ubatch & ubatch) {
    const auto n_kv     = kvcache->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = kv_unified ? 1 : ubatch.n_seqs_unq;

    nnml_tensor_ptrs self_kq_mask_cnvs;
    for (int i = 0; i < kvcache->get_n_para_kvcaches(); i++) {
        kvcache->build_input_k_idxs(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, ubatch);
        kvcache->build_input_v_idxs(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, ubatch, hparams);
        nnml_tensor * self_kq_mask = tensor_new_4d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, NNML_TYPE_F32, n_kv, NNML_PAD(n_tokens/n_stream, NNML_KQ_MASK_PAD), 1, n_stream);
        self_kq_mask->set_name("%s-N%d", "self_kq_mask", i);
        kvcache->set_kq_mask(self_kq_mask, i);
        nnml_tensor * self_kq_mask_cnv = flash_attn ? nnml_cast(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, self_kq_mask, NNML_TYPE_F16) : self_kq_mask;
        if (flash_attn) build_forward_expand(self_kq_mask_cnv);             // serial build
        kvcache->set_kq_mask_cnv(self_kq_mask_cnv, i);
    }
    n_kv_max = n_kv;                                //update the n_kv_max of this graph
}

void nnml_cgraph::build_attn_inp_kv() {
    build_attn_inp_kv_impl(ubatch);
}

nnml_tensor * nnml_cgraph::build_norm(nnml_tensor * cur, nnml_tensor * mw, nnml_tensor * mb, llm_norm_type type, int il) {
    switch (type) {
        case LLM_NORM:       cur = nnml_norm    (mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.get_f_norm_eps());     break;
        case LLM_NORM_RMS:   cur = nnml_rms_norm(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.get_f_norm_rms_eps()); break;
        case LLM_NORM_GROUP:
            {
                cur = nnml_reshape_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, cur->get_elements(0), 1, cur->get_elements(1));
                cur = nnml_group_norm(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.get_n_norm_groups(), hparams.get_f_norm_group_eps());
                cur = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, cur->get_elements(0), cur->get_elements(2));
            } break;
    }
    if (mw || mb) {
        cur->set_name("%s-%d", "norm", il);
    }
    build_forward_expand(cur);
    if (mw) {
        cur = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, mw);
        if (mb) {
            cur->set_name("%s-%d", "norm_w", il);
        }
        build_forward_expand(cur);
    }
    if (mb) {
        cur = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, mb);
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor_ptrs nnml_cgraph::build_norm(nnml_tensor_ptrs cur, nnml_tensor ** mw, nnml_tensor ** mb, llm_norm_type type, int il) {
    nnml_tensor_ptrs res;
    switch (type) {
        case LLM_NORM:       for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_norm(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], hparams.get_f_norm_eps());     break;
        case LLM_NORM_RMS:   for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_rms_norm(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], hparams.get_f_norm_rms_eps());  break;
        case LLM_NORM_GROUP:
            {
                for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_reshape_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], cur[i]->get_elements(0), 1, cur[i]->get_elements(1));
                build_forward_expand(res);
                for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_group_norm(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, res[i], hparams.get_n_norm_groups(), hparams.get_f_norm_group_eps());
                build_forward_expand(res);
                for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, res[i], res[i]->get_elements(0), res[i]->get_elements(2));
            } break;
    }
    if (mw || mb) {
        res.set_name("%s-%d", "norm", il);
    }
    build_forward_expand(res);
    if (mw) {
        for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, res[i], mw[i]);
        if (mb) {
            res.set_name("%s-%d", "norm_w", il);
        }
        build_forward_expand(res);
    }
    if (mb) {
        for (int i = 0; i < n_para_graphs; ++i) res[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, res[i], mb[i]);
        build_forward_expand(res);
    }
    return res;
}

nnml_tensor * nnml_cgraph::build_lora_mm(nnml_tensor * w, nnml_tensor * cur) {
    nnml_tensor * res = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, w, cur);
    if (w->is_asm_gemm()) {
        res->set_asm_gemm();
    }
    // TODO: we do not support lora arch in the current version.
    build_forward_expand(res);
    return res;
}

nnml_tensor_ptrs nnml_cgraph::build_lora_mm(nnml_tensor ** w, nnml_tensor_ptrs cur) {
    nnml_tensor_ptrs res;
    for (int i = 0; i < n_para_graphs; ++i) {
        res[i] = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, w[i], cur[i]);
        if (w[i]->is_asm_gemm()) {
            res[i]->set_asm_gemm();
        }
    }
    build_forward_expand(res);
    return res;
}

nnml_tensor * nnml_cgraph::build_add(nnml_tensor * cur, nnml_tensor * w) {
    nnml_tensor * res = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, w);
    build_forward_expand(res);
    return res;
}

nnml_tensor_ptrs nnml_cgraph::build_add(nnml_tensor_ptrs cur, nnml_tensor_ptrs w) {
    if (n_para_graphs == 1) {
        nnml_tensor_ptrs res = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur[0], w[0]);
        return res;
    } else 
        return nullptr;
}

nnml_tensor * nnml_cgraph::build_reshape_3d(nnml_tensor * cur, int64_t n_head, int64_t n_embd_head, int64_t n_tokens) {
    nnml_tensor * res = nnml_reshape_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, n_head, n_embd_head, n_tokens);
    build_forward_expand(res);
    return res;
}

nnml_tensor_ptrs nnml_cgraph::build_reshape_3d(nnml_tensor_ptrs cur, int64_t n_head, int64_t n_embd_head, int64_t n_tokens) {
    nnml_tensor_ptrs res;
    for (int i = 0; i < n_para_graphs; ++i) {
        res[i] = nnml_reshape_3d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], n_head, n_embd_head, n_tokens);
    }
    build_forward_expand(res);
    return res;
}

nnml_tensor * nnml_cgraph::build_get_rows(nnml_tensor * cur, nnml_tensor * ids) {
    nnml_tensor * res = nnml_get_rows(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, ids);
    build_forward_expand(res);
    return res;
}

nnml_tensor_ptrs nnml_cgraph::build_get_rows(nnml_tensor_ptrs cur, nnml_tensor_ptrs ids) {
    if (n_para_graphs == 1) {
        nnml_tensor_ptrs res = nnml_get_rows(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur[0], ids[0]);
        return res;
    } else
        return nullptr;
}

nnml_tensor * nnml_cgraph::build_rope_ext(nnml_tensor * cur, nnml_tensor * b, nnml_tensor * c, int n_dims, int mode, int n_ctx_orig, float freq_base,
    float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    nnml_tensor * res = nnml_rope_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, b, c,
        n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    build_forward_expand(res);
    return res;
}

nnml_tensor_ptrs nnml_cgraph::build_rope_ext(nnml_tensor_ptrs cur, nnml_tensor * b, nnml_tensor * c, int n_dims, int mode, int n_ctx_orig, float freq_base,
    float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    nnml_tensor_ptrs res;
    for (int i = 0; i < n_para_graphs; ++i) {
        res[i] = nnml_rope_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], b, c,
            n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    build_forward_expand(res);
    return res;
}

nnml_tensor * nnml_cgraph::build_attn_mha(nnml_tensor * q, nnml_tensor * k, nnml_tensor * v, nnml_tensor * kq_b, nnml_tensor * kq_mask, nnml_tensor * sinks,
    nnml_tensor * v_mla, float kq_scale, int il) {
    const bool v_trans = v->get_stride_bytes(1) > v->get_stride_bytes(2);
    const auto n_stream = k->get_elements(3);
    
    q = nnml_view_4d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, q, q->get_elements(0), q->get_elements(1), q->get_elements(2)/n_stream, 
                        n_stream, q->get_stride_bytes(1), q->get_stride_bytes(2), q->get_stride_bytes(3)/n_stream, 0);
    q = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, q, 0, 2, 1, 3);
    k = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, k, 0, 2, 1, 3);
    k->set_name("%s-%d", "cache_k_d", il);
    build_forward_expand(k);
    v = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v, 0, 2, 1, 3);
    v->set_name("%s-%d", "cache_v_d", il);
    build_forward_expand(v);
    const auto n_kv = k->get_elements(1);
    nnml_tensor * cur;

    if (flash_attn && (n_kv % 256 == 0) && kq_b == nullptr) {
        // printf("k type: %d\n", k->get_data_type());
        NNML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
        if (v_trans) {
            v = nnml_transpose(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v);
        }
        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->get_data_type() == NNML_TYPE_F32) {
            k = nnml_cast(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, k, NNML_TYPE_F16);
            build_forward_expand(k);
        }
        if (v->get_data_type() == NNML_TYPE_F32) {
            v = nnml_cast(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v, NNML_TYPE_F16);
            build_forward_expand(v);
        }

        cur = nnml_flash_attn_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, q, k, v, kq_mask, kq_scale, hparams.get_f_max_alibi_bias(),
                                  hparams.get_attn_soft_cap() ? hparams.get_f_attn_logit_softcapping() : 0.0f);
        cur->set_name("%s-%d", LLM_TENSOR_NAME_FATTN, il);
        build_forward_expand(cur);

        nnml_flash_attn_ext_add_sinks(cur, sinks);
        nnml_flash_attn_ext_set_prec(cur, NNML_PREC_F32);
        if (v_mla) {
            cur = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, 0, 2, 1, 3);
            cur = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v_mla, cur);
            cur->set_name("%s-%d", "fattn_mla", il);
            build_forward_expand(cur);
            cur = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, 0, 2, 1, 3);
            cur = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
            build_forward_expand(cur);
        }
        cur = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, cur->get_elements(0)*cur->get_elements(1), cur->get_elements(2)*cur->get_elements(3));
    } else {
        nnml_tensor * kq = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, k, q);
        kq->set_name("%s-%d", "kq", il);
        build_forward_expand(kq);
        nnml_mul_mat_set_prec(kq, NNML_PREC_F32);

        if (hparams.attn_soft_cap) {
            kq = nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq->set_name("%s-%d", "kq_scaled_1", il);
            build_forward_expand(kq);
            kq = nnml_tanh(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kq);
            kq->set_name("%s-%d", "kq_tanh", il);
            build_forward_expand(kq);
            kq = nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kq, hparams.f_attn_logit_softcapping);
            kq->set_name("%s-%d", "kq_scaled_2", il);
            build_forward_expand(kq);
        }

        if (kq_b) {
            kq = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kq, kq_b);
            kq->set_name("%s-%d", "kq_plus_kq_b", il);
            build_forward_expand(kq);
        }

        kq = nnml_soft_max_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        nnml_soft_max_add_sinks(kq, sinks);
        kq->set_name("%s-%d", "kq_soft_max", il);
        build_forward_expand(kq);

        if (!v_trans) {
            // note: avoid this branch
            v = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, nnml_transpose(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v));
            v->set_name("%s-%d", "v_cont", il);
            build_forward_expand(v);
        }
        
        nnml_tensor * kqv = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v, kq);
        kqv->set_name("%s-%d", "kqv", il);
        build_forward_expand(kqv);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v_mla, kqv);
            kqv->set_name("%s-%d", "kqv_mla", il);
            build_forward_expand(kqv);
        }

        cur = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, kqv, 0, 2, 1, 3);
        // recombine streams
        cur = nnml_cont_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, cur->get_elements(0)*cur->get_elements(1), cur->get_elements(2)*cur->get_elements(3));
        build_forward_expand(cur);
    }

    return cur;
}

nnml_tensor_ptrs nnml_cgraph::build_attn_mha(nnml_tensor_ptrs q, nnml_tensor_ptrs k, nnml_tensor_ptrs v, nnml_tensor ** kq_b, nnml_tensor ** kq_mask, nnml_tensor ** sinks,
    nnml_tensor ** v_mla, float kq_scale, int il) {
    const bool v_trans = v[0]->get_stride_bytes(1) > v[0]->get_stride_bytes(2);
    const auto n_stream = k[0]->get_elements(3);
    nnml_tensor_ptrs q_tmp, k_tmp, v_tmp;
    for (int i = 0; i < n_para_graphs; ++i) {
        q_tmp[i] = nnml_view_4d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, q[i], q[i]->get_elements(0), q[i]->get_elements(1), q[i]->get_elements(2)/n_stream, 
                            n_stream, q[i]->get_stride_bytes(1), q[i]->get_stride_bytes(2), q[i]->get_stride_bytes(3)/n_stream, 0);
        q_tmp[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, q_tmp[i], 0, 2, 1, 3);
        k_tmp[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, k[i], 0, 2, 1, 3);
        v_tmp[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v[i], 0, 2, 1, 3);
    }
    k_tmp.set_name("%s-%d", "cache_k_d", il);
    build_forward_expand(k_tmp);
    v_tmp.set_name("%s-%d", "cache_v_d", il);
    build_forward_expand(v_tmp);
    const auto n_kv = k_tmp[0]->get_elements(1);
    nnml_tensor_ptrs cur;

    if (flash_attn && (n_kv % 256 == 0) && kq_b == nullptr) {
        NNML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
        if (v_trans) {
            for (int i = 0; i < n_para_graphs; ++i) v_tmp[i] = nnml_transpose(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_tmp[i]);
        }
        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k_tmp[0]->get_data_type() == NNML_TYPE_F32) {
            for (int i = 0; i < n_para_graphs; ++i) k_tmp[i] = nnml_cast(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, k_tmp[i], NNML_TYPE_F16);
            build_forward_expand(k_tmp);
        }
        if (v_tmp[0]->get_data_type() == NNML_TYPE_F32) {
            for (int i = 0; i < n_para_graphs; ++i) v_tmp[i] = nnml_cast(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_tmp[i], NNML_TYPE_F16);
            build_forward_expand(v_tmp);
        }
        nnml_tensor_ptrs kq_mask_tmp;
        for (int i = 0; i < n_para_graphs; ++i) {
            kq_mask_tmp[i] = (kq_mask != nullptr) ? kq_mask[i] : nullptr;
            cur[i] = nnml_flash_attn_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, q_tmp[i], k_tmp[i], v_tmp[i], kq_mask_tmp[i], kq_scale, hparams.get_f_max_alibi_bias(),
                                    hparams.get_attn_soft_cap() ? hparams.get_f_attn_logit_softcapping() : 0.0f);
        }
        cur.set_name("%s-%d", LLM_TENSOR_NAME_FATTN, il);
        build_forward_expand(cur);

        nnml_tensor_ptrs sinks_tmp;
        for (int i = 0; i < n_para_graphs; ++i) {
            sinks_tmp[i] = (sinks != nullptr) ? sinks[i] : nullptr;
            nnml_flash_attn_ext_add_sinks(cur[i], sinks_tmp[i]);
            nnml_flash_attn_ext_set_prec(cur[i], NNML_PREC_F32);
        }
        if (v_mla) {
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            for (int i = 0; i < n_para_graphs; ++i) {
                cur[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], 0, 2, 1, 3);
                cur[i] = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_mla[i], cur[i]);
            }
            cur.set_name("%s-%d", "fattn_mla", il);
            build_forward_expand(cur);
            for (int i = 0; i < n_para_graphs; ++i) {
                cur[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], 0, 2, 1, 3);
                cur[i] = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i]); // Needed because ggml_reshape_2d expects contiguous inputs.
            }
            build_forward_expand(cur);
        }
        for (int i = 0; i < n_para_graphs; ++i)
            cur[i] = nnml_reshape_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, 
                cur[i], cur[i]->get_elements(0)*cur[i]->get_elements(1), cur[i]->get_elements(2)*cur[i]->get_elements(3));
    } else {
        nnml_tensor_ptrs kq;
        for (int i = 0; i < n_para_graphs; ++i) kq[i] = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, k_tmp[i], q_tmp[i]);
        kq.set_name("%s-%d", "kq", il);
        build_forward_expand(kq);
        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        for (int i = 0; i < n_para_graphs; ++i) nnml_mul_mat_set_prec(kq[i], NNML_PREC_F32);

        if (hparams.attn_soft_cap) {
            for (int i = 0; i < n_para_graphs; ++i) kq[i] = nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kq[i], 1.0f / hparams.f_attn_logit_softcapping);
            kq.set_name("%s-%d", "kq_scaled_1", il);
            build_forward_expand(kq);
            for (int i = 0; i < n_para_graphs; ++i) kq[i] = nnml_tanh(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kq[i]);
            kq.set_name("%s-%d", "kq_tanh", il);
            build_forward_expand(kq);
            for (int i = 0; i < n_para_graphs; ++i) kq[i] = nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kq[i], hparams.f_attn_logit_softcapping);
            kq.set_name("%s-%d", "kq_scaled_2", il);
            build_forward_expand(kq);
        }

        if (kq_b) {
            for (int i = 0; i < n_para_graphs; ++i) kq[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kq[i], kq_b[i]);
            kq.set_name("%s-%d", "kq_plus_kq_b", il);
            build_forward_expand(kq);
        }
        
        nnml_tensor_ptrs kq_mask_tmp, sinks_tmp;
        for (int i = 0; i < n_para_graphs; ++i) {
            kq_mask_tmp[i] = (kq_mask != nullptr) ? kq_mask[i] : nullptr;
            sinks_tmp[i] = (sinks != nullptr) ? sinks[i] : nullptr;
            kq[i] = nnml_soft_max_ext(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kq[i], kq_mask_tmp[i], kq_scale, hparams.f_max_alibi_bias);
            nnml_soft_max_add_sinks(kq[i], sinks_tmp[i]);
        }
        kq.set_name("%s-%d", "kq_soft_max", il);
        build_forward_expand(kq);

        if (!v_trans) {
            // note: avoid this branch
            for (int i = 0; i < n_para_graphs; ++i) v_tmp[i] = nnml_cont(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, nnml_transpose(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_tmp[i]));
            v_tmp.set_name("%s-%d", "v_cont", il);
            build_forward_expand(v_tmp);
        }
        
        nnml_tensor_ptrs kqv;
        for (int i = 0; i < n_para_graphs; ++i) kqv[i] = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_tmp[i], kq[i]);
        kqv.set_name("%s-%d", "kqv", il);
        build_forward_expand(kqv);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            for (int i = 0; i < n_para_graphs; ++i) kqv[i] = nnml_mul_mat(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_mla[i], kqv[i]);
            kqv.set_name("%s-%d", "kqv_mla", il);
            build_forward_expand(kqv);
        }

        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_permute(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, kqv[i], 0, 2, 1, 3);
        // recombine streams
        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_cont_2d(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, 
            cur[i], cur[i]->get_elements(0)*cur[i]->get_elements(1), cur[i]->get_elements(2)*cur[i]->get_elements(3));
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor * nnml_cgraph::build_attn(nnml_tensor * wo, nnml_tensor * wo_b, nnml_tensor * q_cur, nnml_tensor * k_cur, nnml_tensor * v_cur,
    nnml_tensor * kq_b, nnml_tensor * sinks, nnml_tensor * v_mla, float kq_scale, int il) {
    {
        const auto & k_idxs = kvcache->get_k_idxs(0);
        const auto & v_idxs = kvcache->get_v_idxs(0);
        build_forward_expand(kvcache->cpy_k(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, k_cur, k_idxs, il, kvcache->get_slot_info(), this));
        build_forward_expand(kvcache->cpy_v(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, v_cur, v_idxs, il, kvcache->get_slot_info(), this));
    }

    const auto & kq_mask = kvcache->get_kq_mask(0);
    nnml_tensor * q = q_cur;
    nnml_tensor * k = kvcache->get_k(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, il, kvcache->get_n_kv(), kvcache->get_slot_info(), hparams);
    nnml_tensor * v = kvcache->get_v(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, il, kvcache->get_n_kv(), kvcache->get_slot_info(), hparams);
    nnml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cur->set_name("%s-%d", "kqv_out", il);
    if (wo) {
        cur = build_lora_mm(wo, cur);
        cur->set_name("%s-%d", "o_out", il);
        build_forward_expand(cur);
    }
    if (wo_b) {
        cur = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, wo_b);
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor_ptrs nnml_cgraph::build_attn(nnml_tensor ** wo, nnml_tensor ** wo_b, nnml_tensor_ptrs q_cur, nnml_tensor_ptrs k_cur, nnml_tensor_ptrs v_cur,
    nnml_tensor ** kq_b, nnml_tensor ** sinks, nnml_tensor ** v_mla, float kq_scale, int il) {
    nnml_tensor_ptrs resk, resv;
    for (int i = 0; i < n_para_graphs; ++i) {
        const auto & k_idxs = kvcache->get_k_idxs(i);
        const auto & v_idxs = kvcache->get_v_idxs(i);
        resk[i] = kvcache->cpy_k(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, k_cur[i], k_idxs, il, kvcache->get_slot_info(), this);
        resv[i] = kvcache->cpy_v(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, v_cur[i], v_idxs, il, kvcache->get_slot_info(), this);
    }
    build_forward_expand(resk);
    build_forward_expand(resv);
    
    nnml_tensor_ptrs q, k, v, kq_mask;
    for (int i = 0; i < n_para_graphs; ++i) {
        kq_mask[i] = kvcache->get_kq_mask(i);
        q[i] = q_cur[i];
        k[i] = kvcache->get_k(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, il, kvcache->get_n_kv(), kvcache->get_slot_info(), hparams);
        v[i] = kvcache->get_v(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, il, kvcache->get_n_kv(), kvcache->get_slot_info(), hparams);
    }
    
    nnml_tensor_ptrs cur = build_attn_mha(q, k, v, kq_b, kq_mask.get_tensor_ptrs(), sinks, v_mla, kq_scale, il);
    cur.set_name("%s-%d", "kqv_out", il);
    // nnml_tensor_ptrs res;
    if (wo) {
        cur = build_lora_mm(wo, cur);
        cur.set_name("%s-%d", "o_out", il);
    }
    if (wo_b) {
        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], wo_b[i]);
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor * nnml_cgraph::build_ffn(nnml_tensor * cur, nnml_tensor * up, nnml_tensor * up_b, nnml_tensor * up_s, nnml_tensor * gate, nnml_tensor * gate_b, nnml_tensor * gate_s,
    nnml_tensor * down, nnml_tensor * down_b, nnml_tensor * down_s, nnml_tensor * act_scales, llm_ffn_op_type type_op, llm_ffn_gate_type type_gate, int il) {
    nnml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    tmp->set_name("%s-%d", "ffn_up", il);
    if (up_b) {
        tmp = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, tmp, up_b);
        tmp->set_name("%s-%d", "ffn_up_b", il);
        build_forward_expand(tmp);
    }
    if (up_s) {
        tmp = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, tmp, up_s);
        tmp->set_name("%s-%d", "ffn_up_s", il);
        build_forward_expand(tmp);
    }
    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cur->set_name("%s-%d", "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cur->set_name("%s-%d", "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, gate_b);
            cur->set_name("%s-%d", "ffn_gate_b", il);
            build_forward_expand(cur);
        }

        if (gate_s) {
            cur = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, gate_s);
            cur->set_name("%s-%d", "ffn_gate_s", il);
            build_forward_expand(cur);
        }
    } else {
        cur = tmp;
    }
    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = nnml_swiglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, tmp);
                cur->set_name("%s-%d", "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = nnml_silu(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur);
                cur->set_name("%s-%d", "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = nnml_geglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, tmp);
                cur->set_name("%s-%d", "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = nnml_gelu(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur);
                cur->set_name("%s-%d", "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = nnml_div(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, act_scales);
                    cur->set_name("%s-%d", "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = nnml_reglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, tmp);
                cur->set_name("%s-%d", "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = nnml_relu(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur);
                cur->set_name("%s-%d", "ffn_relu", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = nnml_swiglu(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur);
                cur->set_name("%s-%d", "ffn_swiglu", il);
            } break;
        default:
            NNML_ABORT("fatal error");
    }
    build_forward_expand(cur);

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, tmp);
        cur->set_name("%s-%d", "ffn_gate_par", il);
        build_forward_expand(cur);
    }
    if (down) {
        cur = build_lora_mm(down, cur);
    }
    if (down_b) {
        cur->set_name("%s-%d", "ffn_down", il);
    }
    if (down_b) {
        cur = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, down_b);
        build_forward_expand(cur);
    }
    if (down_s) {
        cur = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, down_s);
        cur->set_name("%s-%d", "ffn_down_s", il);
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor_ptrs nnml_cgraph::build_ffn(nnml_tensor_ptrs cur, nnml_tensor ** up, nnml_tensor ** up_b, nnml_tensor ** up_s, nnml_tensor ** gate, nnml_tensor ** gate_b, nnml_tensor ** gate_s,
    nnml_tensor ** down, nnml_tensor ** down_b, nnml_tensor ** down_s, nnml_tensor ** act_scales, llm_ffn_op_type type_op, llm_ffn_gate_type type_gate, int il) {
    nnml_tensor_ptrs tmp = up ? build_lora_mm(up, cur) : cur;
    tmp.set_name("%s-%d", "ffn_up", il);
    if (up_b) {
        for (int i = 0; i < n_para_graphs; ++i) tmp[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, tmp[i], up_b[i]);
        tmp.set_name("%s-%d", "ffn_up_b", il);
        build_forward_expand(tmp);
    }
    if (up_s) {
        for (int i = 0; i < n_para_graphs; ++i) tmp[i] = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, tmp[i], up_s[i]);
        tmp.set_name("%s-%d", "ffn_up_s", il);
        build_forward_expand(tmp);
    }
    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cur.set_name("%s-%d", "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cur.set_name("%s-%d", "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], gate_b[i]);
            cur.set_name("%s-%d", "ffn_gate_b", il);
            build_forward_expand(cur);
        }

        if (gate_s) {
            for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], gate_s[i]);
            cur.set_name("%s-%d", "ffn_gate_s", il);
            build_forward_expand(cur);
        }
    } else {
        cur = tmp;
    }
    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_swiglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], tmp[i]);
                cur.set_name("%s-%d", "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_silu(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i]);
                cur.set_name("%s-%d", "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_geglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], tmp[i]);
                cur.set_name("%s-%d", "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_gelu(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i]);
                cur.set_name("%s-%d", "ffn_gelu", il);
                if (act_scales != NULL) {
                    for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_div(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], act_scales[i]);
                    cur.set_name("%s-%d", "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_reglu_split(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], tmp[i]);
                cur.set_name("%s-%d", "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_relu(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i]);
                cur.set_name("%s-%d", "ffn_relu", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_swiglu(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i]);
                cur.set_name("%s-%d", "ffn_swiglu", il);
            } break;
        default:
            NNML_ABORT("fatal error");
    }
    build_forward_expand(cur);

    if (gate && type_gate == LLM_FFN_PAR) {
        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], tmp[i]);
        cur.set_name("%s-%d", "ffn_gate_par", il);
        build_forward_expand(cur);
    }
    if (down) {
        cur = build_lora_mm(down, cur);
    }
    if (down_b) {
        cur.set_name("%s-%d", "ffn_down", il);
    }
    if (down_b) {
        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_add(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], down_b[i]);
        build_forward_expand(cur);
    }
    if (down_s) {
        for (int i = 0; i < n_para_graphs; ++i) cur[i] = nnml_mul(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, cur[i], down_s[i]);
        cur.set_name("%s-%d", "ffn_down_s", il);
        build_forward_expand(cur);
    }
    return cur;
}

nnml_tensor_ptrs nnml_cgraph::build_scatter(nnml_tensor * src, bool is_by_row) {
    nnml_tensor * cur = nnml_scatter_prepare(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, src);          // reorganize the thread pool
    build_forward_expand(cur);
    NNML_UNUSED(is_by_row);

    // turn to scatter-build mode
    n_para_graphs = n_para_graphs_max;
    is_para_mode = false;

    nnml_tensor_ptrs res;
    for (int i = 0; i < n_para_graphs; ++i) {
        // only 2-D weight tensor supported
        nnml_tensor * child = nnml_scatter_copy(mem, NNML_TENSOR_TYPE_ACTIVATION, i, 0, src);
        // child->set_name("%s-N%d", "scatter", i);
        res[i] = child;
    }

    build_forward_expand(res);
    // turn to parallel-build mode
    is_para_mode = true;

    return res;
}

nnml_tensor * nnml_cgraph::build_gather(nnml_tensor_ptrs src, bool is_by_row) {
    NNML_UNUSED(is_by_row);
    nnml_tensor ** tensor_ptrs = src.get_tensor_ptrs();
    // only 2-D weight tensor supported
    nnml_tensor * cur = nnml_gather_copy(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, tensor_ptrs, n_para_graphs);
    nnml_tensor_ptrs cur_ptrs = cur;

    // turn to gather-build mode
    n_para_graphs = 1;
    is_para_mode = true;
    build_forward_expand(cur_ptrs);
    // turn to serial-build mode
    is_para_mode = false;
    return cur;
}

void nnml_cgraph::build_forward_impl(nnml_tensor * tensor, bool expand) {
    // here we use the simplest implementation: collect the current node into nodes array, to build the topological order
    if(n_para_graphs == 1 && is_para_mode == false) {   // only one child
        nodes[n_nodes++].tensor = tensor;
        if (n_nodes > 1) nodes[n_nodes-2].child[0] = n_nodes - 1;
    } else {                                            // multiple children in para mode
        abort();
    }
}

void nnml_cgraph::build_forward_impl(nnml_tensor_ptrs tensor, bool expand) {
    NNML_UNUSED(expand);
    if (n_nodes + n_para_graphs >= n_cnode_array) {
        NNML_ABORT("fatal error: cgraph nodes overflow, "
                   "n_nodes=%d, n_para_graphs=%d, n_cnode_array=%d",
                   n_nodes, n_para_graphs, n_cnode_array);
    }
    const bool serial_mode   = (n_para_graphs == 1) && !is_para_mode;
    const bool scatter_mode  = (n_para_graphs >  1) && !is_para_mode;
    const bool parallel_mode = (n_para_graphs >  1) &&  is_para_mode;
    const bool gather_mode   = (n_para_graphs == 1) &&  is_para_mode;

    if (serial_mode) {
        nodes[n_nodes].tensor = tensor;
        if (n_nodes > 0) {
            nodes[n_nodes - 1].child[0] = n_nodes;
        }
        ++n_nodes;
        return;
    }
    if (scatter_mode) {
        for (int32_t i = 0; i < n_para_graphs_max; ++i) {
            nodes[n_nodes + i].tensor = tensor[i];
            nodes[n_nodes - 1].child[i] = n_nodes + i;
        }
        n_nodes += n_para_graphs;
        return;
    }
    if (parallel_mode) {
        for (int32_t i = 0; i < n_para_graphs_max; ++i) {
            nodes[n_nodes + i].tensor = tensor[i];
            nodes[n_nodes + i - n_para_graphs_max].child[i] = n_nodes + i;
        }
        n_nodes += n_para_graphs;
        return;
    }
    if (gather_mode) {
        nodes[n_nodes].tensor = tensor;
        for (int32_t i = 0; i < n_para_graphs_max; ++i) {
            nodes[n_nodes + i - n_para_graphs_max].child[i] = n_nodes;
        }
        ++n_nodes;
        return;
    }
    NNML_ABORT("fatal error: n_para_graphs=%d is_para_mode=%d", n_para_graphs, is_para_mode ? 1 : 0);
}

void nnml_cgraph::build_forward_expand(nnml_tensor * tensor) {
    build_forward_impl(tensor, true);
}

void nnml_cgraph::build_forward_expand(nnml_tensor_ptrs tensor) {
    build_forward_impl(tensor, true);
}

void nnml_cgraph::set_inputs_embd(const llm_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;
        t_tokens->copy_from(ubatch->token, 0, n_tokens*nnml_type_size(t_tokens->get_data_type()));
    }
    if (ubatch->embd) {
        const int64_t n_embd   = t_embd->get_elements(0);
        const int64_t n_tokens = ubatch->n_tokens;
        t_embd->copy_from(ubatch->embd, 0, n_tokens*n_embd*nnml_type_size(t_embd->get_data_type()));
    }
}

void nnml_cgraph::set_inputs_pos(const llm_ubatch * ubatch) {
    if (ubatch->pos && t_pos) {
        const int64_t n_tokens = ubatch->n_tokens;
        if (ubatch->token && hparams.n_pos_per_embd() == 4) {
            std::vector<llm_token> pos_data(n_tokens*hparams.n_pos_per_embd());
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            t_pos->copy_from(pos_data.data(), 0, pos_data.size()*nnml_type_size(t_pos->get_data_type()));
        } else {
            t_pos->copy_from(ubatch->pos, 0, n_tokens*hparams.n_pos_per_embd()*nnml_type_size(t_pos->get_data_type()));
        }
    }
}

void nnml_cgraph::set_inputs_out_ids(const llm_ubatch * ubatch) {
    // we simply implement a one-batch edition
    const int64_t n_tokens = ubatch->n_tokens;
    int32_t * data = (int32_t *) t_out_ids->tensor_data();
    if (n_tokens > 1) data[0] = n_tokens - 1;
    else data[0] = 0;
}

void nnml_cgraph::set_inputs_attn_kv(const llm_ubatch * ubatch) {
    kvcache->set_inputs_attn_kidx(ubatch);
    kvcache->set_inputs_attn_vidx(ubatch, hparams);
    kvcache->set_inputs_attn_mask(ubatch);
}

void nnml_cgraph::set_inputs(const llm_ubatch * ubatch) {
    set_inputs_embd(ubatch);
    set_inputs_pos(ubatch);
    set_inputs_out_ids(ubatch);
    set_inputs_attn_kv(ubatch);
}


// implementation of llm_kv_cache
llm_kv_cache::llm_kv_cache(
        nnml_type        type_k,
        nnml_type        type_v,
        bool             v_trans,
        bool             unified,
        uint32_t         kv_size,
        uint32_t         n_seq_max,
        uint32_t         n_pad,
        llm_hparams      hparams,
        nnml_memory_t &  mem,
        int32_t          n_para_kvcaches,
        bool             is_print_kv
    )
    : v_trans(v_trans), n_seq_max(n_seq_max), n_stream(unified ? 1u : n_seq_max), n_pad(n_pad),
      n_para_kvcaches(n_para_kvcaches), n_layer(hparams.n_layer), i_cur(0)
{   
    NNML_ASSERT(kv_size % n_pad == 0);
    NNML_ASSERT(n_stream == 1 || n_stream == n_seq_max);
    constexpr nnml_tensor_type tensor_type = NNML_TENSOR_TYPE_KVCACHE;

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }
    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }
    seq_to_stream.resize(LLM_MAX_SEQ, 0);       // by default, all sequence ids are mapped to the 0th stream
    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        NNML_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            NNML_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
        // WARNING: we only assume that n_embd_k_gqa and n_embd_v_gqa are divisible by n_para_kvcaches, but this is inconsistent with judging elsewhere
        NNML_ASSERT(n_embd_k_gqa % n_para_kvcaches == 0);
        NNML_ASSERT(n_embd_v_gqa % n_para_kvcaches == 0);

        kv_layer new_layer;
        new_layer.il = il;
        for (uint32_t i = 0; i < n_para_kvcaches; ++i) {
            new_layer.k[i] = tensor_new_3d(mem, tensor_type, i, 0, type_k, n_embd_k_gqa / n_para_kvcaches, kv_size, n_stream);
            new_layer.v[i] = tensor_new_3d(mem, tensor_type, i, 0, type_v, n_embd_v_gqa / n_para_kvcaches, kv_size, n_stream);
            new_layer.k[i]->set_name("cache_k-%d-N%d", il, i);
            new_layer.v[i]->set_name("cache_v-%d-N%d", il, i);
        }
        map_layer_ids[il] = layers.size();
        layers.push_back(std::move(new_layer));
    }

    if (is_print_kv) {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();
        NNML_LOG("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                nnml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                nnml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }
}

void llm_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }
}

void llm_kv_cache::init_batch(const llm_ubatch & ubatch) {      // this func is not used in this version
    slot_info res = {
        /*.s0   =*/ 0,
        /*.s1   =*/ 0,
        /*.strm =*/ {},
        /*.idxs =*/ {},
    };
    res.strm.resize(1);
    res.idxs.resize(1);
    sinfos.clear();
    sinfos.push_back(res);
}

llm_pos llm_kv_cache::seq_pos_min(llm_seq_id seq_id) const {
    NNML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.seq_pos_min(seq_id);
}

llm_pos llm_kv_cache::seq_pos_max(llm_seq_id seq_id) const {
    NNML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.seq_pos_max(seq_id);
}

size_t llm_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;
    for (const auto & layer : layers) {
        for (int i = 0; i < n_para_kvcaches; ++i) size_k_bytes += nnml_nbytes(layer.k[i]);
    }
    return size_k_bytes;
}

size_t llm_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;
    for (const auto & layer : layers) {
        for (int i = 0; i < n_para_kvcaches; ++i) size_v_bytes += nnml_nbytes(layer.v[i]);
    }
    return size_v_bytes;
}

uint32_t llm_kv_cache::get_size() const {
        const auto & cells = v_cells[seq_to_stream[0]];
        return cells.size();
    }

void llm_kv_cache::build_input_k_idxs(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const llm_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;
    self_k_idxs[buffer_id] = tensor_new_1d(mem, tensor_type, buffer_id, dual_idx, NNML_TYPE_I64, n_tokens);
}

void llm_kv_cache::build_input_v_idxs(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const llm_ubatch & ubatch, llm_hparams& hparams) {
    const uint32_t n_tokens = ubatch.n_tokens;
    nnml_tensor * v_idxs;
    if (!v_trans) {
        v_idxs = tensor_new_1d(mem, tensor_type, buffer_id, dual_idx, NNML_TYPE_I64, n_tokens);
    } else {
        v_idxs = tensor_new_1d(mem, tensor_type, buffer_id, dual_idx, NNML_TYPE_I64, n_tokens * hparams.n_embd_v_gqa_max());
    }
    self_v_idxs[buffer_id] = v_idxs;
}

nnml_tensor * llm_kv_cache::get_k(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, int32_t il, uint32_t n_kv, const slot_info & sinfo, llm_hparams & hparams) {
    const int32_t ikv = map_layer_ids.at(il);
    auto * k = layers[ikv].k[buffer_id];
    const uint64_t kv_size      = get_size();
    uint64_t n_embd_k_gqa = k->get_elements(0);

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    // for tensor parallel
    assert(hparams.n_head_kv(il) % n_para_kvcaches == 0);
    int64_t n_head_kv = hparams.n_head_kv(il) / n_para_kvcaches;
    n_embd_k_gqa = n_head_kv * hparams.n_embd_head_k;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    // printf("get_k: il=%d, ikv=%d, n_embd_k_gqa=%lu, kv_size=%lu, ns=%u\n", il, ikv, n_embd_k_gqa, kv_size, ns);
    return nnml_view_4d(mem, tensor_type, buffer_id, dual_idx, k,
            hparams.n_embd_head_k, n_head_kv, n_kv, ns,
            nnml_row_size(k->get_data_type(), hparams.n_embd_head_k),
            nnml_row_size(k->get_data_type(), n_embd_k_gqa),
            nnml_row_size(k->get_data_type(), n_embd_k_gqa*kv_size),
            nnml_row_size(k->get_data_type(), n_embd_k_gqa*kv_size)*sinfo.s0);
}

nnml_tensor * llm_kv_cache::get_v(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, int32_t il, uint32_t n_kv, const slot_info & sinfo, llm_hparams & hparams) {
    const int32_t ikv = map_layer_ids.at(il);
    auto * v = layers[ikv].v[buffer_id];
    const uint64_t kv_size      = get_size();
    uint64_t n_embd_v_gqa = v->get_elements(0);

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // for tensor parallel
    assert(hparams.n_head_kv(il) % n_para_kvcaches == 0);
    int64_t n_head_kv = hparams.n_head_kv(il) / n_para_kvcaches;
    n_embd_v_gqa = n_head_kv * hparams.n_embd_head_v;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return nnml_view_4d(mem, tensor_type, buffer_id, dual_idx, v,
                hparams.n_embd_head_v, n_head_kv, n_kv, ns,
                nnml_row_size(v->get_data_type(), hparams.n_embd_head_v),          // v->nb[1]
                nnml_row_size(v->get_data_type(), n_embd_v_gqa),                   // v->nb[2]
                nnml_row_size(v->get_data_type(), n_embd_v_gqa*kv_size),           // v->nb[3]
                nnml_row_size(v->get_data_type(), n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return nnml_view_4d(mem, tensor_type, buffer_id, dual_idx, v,
            n_kv, n_head_kv, hparams.n_embd_head_v, ns,
            nnml_row_size(v->get_data_type(), kv_size*hparams.n_embd_head_v),  // v->nb[1]
            nnml_row_size(v->get_data_type(), kv_size),                        // v->nb[2]
            nnml_row_size(v->get_data_type(), kv_size*n_embd_v_gqa),           // v->nb[3]
            nnml_row_size(v->get_data_type(), kv_size*n_embd_v_gqa)*sinfo.s0);
}

nnml_tensor * llm_kv_cache::cpy_k(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
                                  nnml_tensor * k_cur, nnml_tensor * k_idxs, int32_t il, const slot_info & sinfo, nnml_cgraph * cgraph) {
    NNML_UNUSED(sinfo);
    const int32_t ikv = map_layer_ids.at(il);
    nnml_tensor * k = layers[ikv].k[buffer_id];
    const int64_t n_embd_head = k_cur->get_elements(0);
    const int64_t n_head      = k_cur->get_elements(1);
    const int64_t n_tokens    = k_cur->get_elements(2);
    const int64_t n_embd_gqa = n_embd_head*n_head;

    NNML_ASSERT(nnml_row_size(k_cur->get_data_type(), n_embd_head) == k_cur->get_stride_bytes(1));
    k_cur = nnml_view_2d(mem, tensor_type, buffer_id, dual_idx, k_cur, n_embd_gqa, n_tokens, k_cur->get_stride_bytes(2), 0);
    const int64_t n_stream = k->get_elements(2);

    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        assert(n_embd_gqa == k->get_elements(0));
        assert(kv_size    == k->get_elements(1));
        // merge the buffer across all streams because the idxs are global
        k = nnml_reshape_2d(mem, tensor_type, buffer_id, dual_idx, k, n_embd_gqa, kv_size*n_stream);
    }
    // store the current K values into the cache
    return nnml_set_rows(mem, tensor_type, buffer_id, dual_idx, k, k_cur, k_idxs);
}

nnml_tensor * llm_kv_cache::cpy_v(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
                                  nnml_tensor * v_cur, nnml_tensor * v_idxs, int32_t il, const slot_info & sinfo, nnml_cgraph * cgraph) {
    NNML_UNUSED(sinfo);
    const int32_t ikv = map_layer_ids.at(il);
    auto * v = layers[ikv].v[buffer_id];
    const int64_t n_embd_head = v_cur->get_elements(0);
    const int64_t n_head      = v_cur->get_elements(1);
    const int64_t n_tokens    = v_cur->get_elements(2);
    const int64_t n_embd_gqa = n_embd_head*n_head;

    NNML_ASSERT(nnml_row_size(v_cur->get_data_type(), n_embd_head) == v_cur->get_stride_bytes(1));
    const int64_t n_stream = v->get_elements(2);
    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = nnml_view_2d(mem, tensor_type, buffer_id, dual_idx, v_cur, n_embd_gqa, n_tokens, v_cur->get_stride_bytes(2), 0);
        if (n_stream > 1) {
            const int64_t kv_size = get_size();
            assert(n_embd_gqa == v->get_elements(0));
            assert(kv_size    == v->get_elements(1));
            // merge the buffer across all streams because the idxs are global
            v = nnml_reshape_2d(mem, tensor_type, buffer_id, dual_idx, v, n_embd_gqa, kv_size*n_stream);
        }
        return nnml_set_rows(mem, tensor_type, buffer_id, dual_idx, v, v_cur, v_idxs);
    }
    if (nnml_row_size(v_cur->get_data_type(), n_embd_gqa) == v_cur->get_stride_bytes(2)) {
        // we can merge dims 0, 1 and 2
        v_cur = nnml_reshape_2d(mem, tensor_type, buffer_id, dual_idx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = nnml_cont_2d(mem, tensor_type, buffer_id, dual_idx, v_cur, n_embd_gqa, n_tokens);
        cgraph->build_forward_expand(v_cur);
    }
    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->get_elements(0)) {
        v_cur = nnml_pad(mem, tensor_type, buffer_id, dual_idx, v_cur, v->get_elements(0) - n_embd_gqa, 0, 0, 0);
        cgraph->build_forward_expand(v_cur);
    }
    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    nnml_tensor * v_view = nnml_reshape_2d(mem, tensor_type, buffer_id, dual_idx, v, 1, v->n_elements());
    v_cur = nnml_reshape_2d(mem, tensor_type, buffer_id, dual_idx, v_cur, 1, v_cur->n_elements());
    return nnml_set_rows(mem, tensor_type, buffer_id, dual_idx, v_view, v_cur, v_idxs);
}

void llm_kv_cache::set_inputs_attn_kidx(const llm_ubatch * ubatch) {
    const uint32_t n_tokens = ubatch->n_tokens;
    NNML_ASSERT(n_tokens == (int64_t) sinfos[i_cur].size()*sinfos[i_cur].n_stream());
    for (int i = 0; i < n_para_kvcaches; ++i) {
        int64_t * data = (int64_t *) self_k_idxs[i]->tensor_data();
        for (uint32_t s = 0; s < sinfos[i_cur].n_stream(); ++s) {
            const int64_t offs = sinfos[i_cur].strm[s]*get_size();
            for (uint32_t i = 0; i < sinfos[i_cur].size(); ++i) {
                data[s*sinfos[i_cur].size() + i] = offs + sinfos[i_cur].idxs[s][i];
            }
        }
    }
}

void llm_kv_cache::set_inputs_attn_vidx(const llm_ubatch * ubatch, llm_hparams& hparams) {
    const uint32_t n_tokens = ubatch->n_tokens;
    NNML_ASSERT(n_tokens == (int64_t) sinfos[i_cur].size()*sinfos[i_cur].n_stream());
    for (int i = 0; i < n_para_kvcaches; ++i) {
        int64_t * data = (int64_t *) self_v_idxs[i]->tensor_data();
        if (!v_trans) {
            for (uint32_t s = 0; s < sinfos[i_cur].n_stream(); ++s) {
                const int64_t offs = sinfos[i_cur].strm[s]*get_size();
                for (uint32_t i = 0; i < sinfos[i_cur].size(); ++i) {
                    data[s*sinfos[i_cur].size() + i] = offs + sinfos[i_cur].idxs[s][i];
                }
            }
        } else {
            // note: the V cache is transposed when not using flash attention
            const int64_t kv_size = get_size();
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();
            for (uint32_t s = 0; s < sinfos[i_cur].n_stream(); ++s) {
                const int64_t offs = sinfos[i_cur].strm[s]*kv_size*n_embd_v_gqa;
                for (uint32_t i = 0; i < sinfos[i_cur].size(); ++i) {
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        data[s*sinfos[i_cur].size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfos[i_cur].idxs[s][i];
                    }
                }
            }
        }
    }
}

void llm_kv_cache::set_inputs_attn_mask(const llm_ubatch * ubatch) {
    for (int i = 0; i < n_para_kvcaches; ++i) {
        const uint32_t n_tokens = ubatch->n_tokens;
        float * data = (float *) self_kq_mask[i]->tensor_data();
        const int64_t n_kv     = self_kq_mask[i]->get_elements(0);
        const int64_t n_stream = self_kq_mask[i]->get_elements(3);          // num streams in the current ubatch
        NNML_ASSERT(n_tokens%n_stream == 0);
        // n_tps == n_tokens_per_stream
        const int64_t n_tps     = n_tokens/n_stream;
        const int64_t n_tps_pad = NNML_PAD(n_tps, NNML_KQ_MASK_PAD);
        std::fill(data, data + self_kq_mask[i]->n_elements(), -INFINITY);
        
        for (uint32_t s = 0; s < n_stream; ++s) {                           // traverse each stream
            for (uint32_t ii = 0; ii < n_tps; ++ii) {                       // traverse each token in the stream
                const uint32_t i = s*n_tps + ii;                            // global index of the token in the stream
                const int32_t seq_id = ubatch->seq_id[i][0];                // get the sequence ID of the token
                const auto & cells = v_cells[seq_to_stream[seq_id]];        // get the cells for the corresponding stream
                const int32_t p1 = ubatch->pos[i];                          // get the position of the token in the sequence
                const uint64_t idst = n_kv*(0 + s*n_tps_pad + ii);          // starting offset in the mask tensor for the current QueryToken
                for (uint32_t j = 0; j < n_kv; ++j) {                       // traverse each token position in the KV cache
                    if (cells.is_empty(j)) {                                // if the physical cache position is empty, skip
                        continue;
                    }
                    // mask the token if not the same sequence
                    if (!cells.seq_has(j, seq_id)) {                        // if the physical cache position is not bound to the current sequence, skip
                        continue;
                    }
                    const int32_t p0 = cells.pos_get(j);                    // get the position of the token in the sequence for the physical cache position
                    // mask future tokens
                    if (p0 > p1) {                                          // if the token position in the physical cache is after the current token position, skip
                        continue;
                    }
                    data[idst + j] = 0.0f;                                  // set the mask value for idst+j to 0, indicating that attention is allowed to access this position
                }
            }
        }
    }
}

std::vector<slot_info> llm_kv_cache::prepare(const std::vector<llm_ubatch> & ubatches) {
    std::vector<slot_info> res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch
        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch
        std::vector<llm_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;
    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }
        // remember the position that we found
        res.push_back(sinfo_new);
        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };
            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];
                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }
            states.push_back(std::move(state));
        }
        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    NNML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];
            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }
    return res;
}

slot_info llm_kv_cache::find_slot(const llm_ubatch & ubatch, bool cont) const {
    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        NNML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);
        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLM_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };
    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            NNML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            NNML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);
        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];
        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            NNML_ABORT("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }
            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;
                head_cur++;
                n_tested++;
                
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llm_pos pos_cell = cells.pos_get(idx);

                    if (!can_use) {
                        const llm_seq_id seq_id_cell = cells.seq_get(idx);
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                return { };
            }
        }
        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);
    return res;
}

void llm_kv_cache::apply_ubatch(const slot_info & sinfo, const llm_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llm_seq_id seq_pos_max_rm[LLM_MAX_SEQ];
    for (uint32_t s = 0; s < LLM_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;
            auto & cells = v_cells[sinfo.strm[s]];
            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);
                const llm_seq_id seq_id = cells.seq_get(idx);
                const llm_pos    pos    = cells.pos_get(idx);
                seq_pos_max_rm[seq_id]  = std::max(seq_pos_max_rm[seq_id], pos);
                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    for (uint32_t s = 0; s < LLM_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }
        NNML_ASSERT(s < seq_to_stream.size());
        auto & cells = v_cells[seq_to_stream[s]];
        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            NNML_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);
            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

const llm_ubatch & llm_kv_cache::get_ubatch() const {
    return ubatches[i_cur];
}

uint32_t llm_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];
        result = std::max(std::min(cells.size(), std::max(n_pad, NNML_PAD(cells.used_max_p1(), n_pad))), result);
    }
    return result;
}

bool llm_kv_cache::apply() {
    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        assert(0);              // not implemented
    }
    apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = get_n_kv(sinfos[i_cur]);
    return true;
}

bool llm_kv_cache::next() {
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    return true;
}

void llm_kv_cache::seq_add(llm_seq_id seq_id, llm_pos p0, llm_pos p1, llm_pos shift) {
    NNML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];
    if (shift == 0) {
        return;
    }
    uint32_t new_head = cells.size();
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llm_pos>::max();
    }
    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }
        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }
    head = new_head != cells.size() ? new_head : 0;
}

bool llm_kv_cache::seq_rm(llm_seq_id seq_id, llm_pos p0, llm_pos p1) {
    NNML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llm_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }
            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}
