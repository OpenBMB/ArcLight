/**
 * @file cgraph.h
 * @brief computation graph builder for NNML.
 * 
 * Provides:
 *  - nnml_cgraph class for building computation graphs from model definition.
 *  - Classes of mini-batch, hyperparameters, and cgraph nodes to facilitate graph building.
 *  - Auxiliary class that encapsulates the bundle of tensor pointers for parallel graph building.
 *  - KV cache definition and management and other utilities related to graph building. (Ack. llama.cpp)
 */
#pragma once

#include <unordered_map>
#include <bitset>
#include <map>
#include <set>
#include <functional>

#include "nnml.h"
#include "tensor.h"
#include "memory.h"


#define LLM_MAX_LAYERS  512
#define LLM_MAX_EXPERTS 384  // Kimi-K2
#define LLM_MAX_SEQ     64

#define NNML_KQ_MASK_PAD 64

#define NNML_MAX_CGRAPH_CNODE 0x0FFFFFFF

#define LLM_TENSOR_NAME_FATTN "__fattn__"



enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

enum llm_rope_type {
    LLM_ROPE_TYPE_NONE   = -1,
    LLM_ROPE_TYPE_NORM   = 0,
    LLM_ROPE_TYPE_NEOX   = 2,
    LLM_ROPE_TYPE_MROPE  = 8,
};

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR,
};

enum llm_expert_gating_func_type {
    LLM_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLM_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3,
};

using llm_token  = int32_t;
using llm_seq_id = int32_t;
using llm_pos    = int32_t;

/**
 * llm_ubatch: unified batch structure for llm inference.
 */
struct llm_ubatch {
    bool equal_seqs() const {
        return b_equal_seqs != 0;
    }

    uint32_t b_equal_seqs; // note: this is a boolean, but we use an int32_t for alignment
                           //       otherwise address sanitizer complains

    uint32_t n_tokens;     // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence set
    uint32_t n_seqs;       // sequence sets in the ubatch
    uint32_t n_seqs_unq;   // unique sequence ids in the ubatch

    //                          // size               | idx | val
    llm_token    *  token;      // [n_tokens]         | i   | id, token
    float        *  embd;       // [n_embd, n_tokens] | i   | embd
    llm_pos      *  pos;        // [n_tokens]         | i   | pos
    int32_t      *  n_seq_id;   // [n_tokens]         | i   | -
    llm_seq_id   ** seq_id;     // [n_tokens]         | s   | s0, s1, seq_id
    llm_seq_id   *  seq_id_unq; // [n_seqs_unq]       | s   | seq_id
    int32_t      *  seq_idx;    // [LLM_MAX_SEQ]      | -   | seq_idx
    int8_t       *  output;     // [n_tokens]         | i   | -

    struct data_t {
        std::vector<llm_token>      token;
        std::vector<float>          embd;
        std::vector<llm_pos>        pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llm_seq_id *>   seq_id;
        std::vector<llm_seq_id>     seq_id_unq;
        std::vector<int32_t>        seq_idx;
        std::vector<int8_t>         output;
    };

    // the llm_ubatch pointers above point to this data if set. otherwise - points to non-owning data
    std::shared_ptr<data_t> data;
};

/**
 * llm_lora_weight: encapsulates the pair of tensors for LoRA weights (A and B).
 *  - This class is not used in the current version.
 *  - However, we leave it here for better modularity and future support of peft models.
 */
struct llm_lora_weight {
    nnml_tensor * a = nullptr;
    nnml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->get_elements(0);
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }
    llm_lora_weight() = default;
    llm_lora_weight(nnml_tensor * a, nnml_tensor * b) : a(a), b(b) {}
};

/**
 * llm_hparams: encapsulates the hyperparameters of the LLM model.
 *  - This class is designed to be easily extendable for future hyperparameters.
 *  - It will not provides setter functions for all hyperparameters.
 *  - Because they are set only once at the beginning of graph building, and
 *  - we assume that the users of llm_hparams know that they are read-only after initialization.
 */
struct llm_hparams {
    uint32_t get_n_embd()                   const noexcept { return n_embd; }
    uint32_t get_n_layer()                  const noexcept { return n_layer; }
    uint32_t get_n_ctx_train()              const noexcept { return n_ctx_train; }
    uint32_t get_n_embd_head_k()            const noexcept { return n_embd_head_k; }
    uint32_t get_n_embd_head_v()            const noexcept { return n_embd_head_v; }
    uint32_t get_n_expert()                 const noexcept { return n_expert; }
    uint32_t get_n_expert_used()            const noexcept { return n_expert_used; }
    uint32_t get_n_norm_groups()            const noexcept { return n_norm_groups; }
    float    get_f_residual_scale()         const noexcept { return f_residual_scale; }
    float    get_f_embedding_scale()        const noexcept { return f_embedding_scale; }
    float    get_f_attention_scale()        const noexcept { return f_attention_scale; }
    float    get_f_norm_eps()               const noexcept { return f_norm_eps; }
    float    get_f_norm_rms_eps()           const noexcept { return f_norm_rms_eps; }
    float    get_f_norm_group_eps()         const noexcept { return f_norm_group_eps; }
    float    get_f_max_alibi_bias()         const noexcept { return f_max_alibi_bias; }
    float    get_f_attn_logit_softcapping() const noexcept { return f_attn_logit_softcapping; }
    bool     get_attn_soft_cap()            const noexcept { return attn_soft_cap; }
    llm_rope_type get_rope_type()           const noexcept { return rope_type; }

    uint32_t n_pos_per_embd() const {
        return rope_type == LLM_ROPE_TYPE_MROPE ? 4 : 1;
    }
    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }
        NNML_ABORT("fatal error");
    }
    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) return n_head_kv_arr[il];
        NNML_ABORT("fatal error");
    }
    uint32_t n_embd_k_gqa(uint32_t il = 0) const {
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_k * n_head_kv;
    }
    uint32_t n_embd_v_gqa(uint32_t il = 0) const {
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_v * n_head_kv;
    }
    uint32_t n_embd_v_gqa_max() const {
        uint32_t val = n_embd_v_gqa();
        for (uint32_t il = 0; il < n_layer; ++il) val = std::max(val, n_embd_v_gqa(il));
        return val;
    }
    bool is_n_embd_v_gqa_variable() const {
        const uint32_t val = n_embd_v_gqa();
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (val != n_embd_v_gqa(il)) {
                return true;
            }
        }
        return false;
    }
    bool has_kv(uint32_t il) const { return true;}

    uint32_t n_embd                 = 512;
    uint32_t n_layer                = 32;
    uint32_t n_ctx_train            = 1024;
    // dimension of keys (d_k). d_q is assumed to be the same,
    // but there are n_head q heads, and only n_head_kv kv heads
    uint32_t n_embd_head_k          = 128;
    // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_v          = 128;
    uint32_t n_expert               = 0;
    uint32_t n_expert_used          = 0;
    uint32_t n_norm_groups          = 1;
    uint32_t n_intermediate_size    = 0;
    int32_t  n_layer_kv_from_start  = -1;
    int64_t  n_rot                  = 0;

    float    f_residual_scale       = 0.0;
    float    f_embedding_scale      = 0.0;
    float    f_attention_scale      = 0.0;
    float    f_norm_eps             = 1e-5;
    float    f_norm_rms_eps         = 1e-6;
    float    f_norm_group_eps       = 1e-5;
    float    f_max_alibi_bias       = 0.0;
    float    f_attn_logit_softcapping = 0.0;
    float    f_rope_freq_base_train = 10000.0;
    float    f_freq_base            = 10000.0;
    float    f_freq_scale           = 1.0;
    float    f_ext_factor           = 1.0;
    float    f_attn_factor          = 1.0;
    float    f_beta_fast            = 0.0;
    float    f_beta_slow            = 0.0;
    bool     attn_soft_cap          = false;

    llm_rope_type rope_type         = LLM_ROPE_TYPE_NONE;

    std::array<uint32_t, LLM_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLM_MAX_LAYERS> n_head_kv_arr;
};

class llm_kv_cells;
class nnml_cgraph;
using idx_vec_t = std::vector<uint32_t>;


/**
 * llm_kv_cache: manages the KV cache for LLM inference.
 *  - The KV cache is organized in slots, which is managed by the llm_kv_cells class.
 *  - llm_kv_cache provides the interfaces for managing the KV tensors, preparing the kv indices for KV cache, and applying the KV cache in graph building.
 *  - It can also build kv_mask for self-attention.
 */
class llm_kv_cache {
public:
    // llm_kv_cache() noexcept = default;
    ~llm_kv_cache() = default;
    // we need implement a proper constructor to initialize the cache
    llm_kv_cache(nnml_type      type_k,    nnml_type        type_v,       bool     v_trans,   bool        unified,
                 uint32_t       kv_size,   uint32_t         n_seq_max,    uint32_t n_pad,     llm_hparams hparams,
                 nnml_memory_t& mem,       int32_t   n_para_kvcaches = 1, bool     is_print_kv = false);
    
    struct slot_info {
        uint32_t s0;                  // number of streams: ns = s1 - s0 + 1
        uint32_t s1;
        std::vector<llm_seq_id> strm; // [ns]
        std::vector<idx_vec_t>  idxs; // [ns]

        uint32_t head() const {
            NNML_ASSERT(idxs.size() == 1);
            NNML_ASSERT(!idxs[0].empty());
            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            NNML_ASSERT(idxs.size() == strm.size());
            NNML_ASSERT(!idxs.empty());
            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }
    };

    slot_info & get_slot_info() {
        NNML_ASSERT(i_cur < sinfos.size());
        return sinfos[i_cur];
    }

    void                    init_batch(const llm_ubatch & ubatch);
    std::vector<slot_info>  prepare(const std::vector<llm_ubatch> & ubatches);
    slot_info               find_slot(const llm_ubatch & ubatch, bool cont) const;
    bool                    apply();
    void                    apply_ubatch(const slot_info & sinfo, const llm_ubatch & ubatch);
    const llm_ubatch &      get_ubatch() const;
    void                    set_ubatchs(const std::vector<llm_ubatch> & ubatches_) { ubatches = std::move(ubatches_); }
    bool                    next();
    void                    set_sinfos(const std::vector<slot_info> & sinfos_) { sinfos = std::move(sinfos_); }

    void    seq_add(llm_seq_id seq_id, llm_pos p0, llm_pos p1, llm_pos shift);
    bool    seq_rm(llm_seq_id seq_id, llm_pos p0, llm_pos p1);
    llm_pos seq_pos_min(llm_seq_id seq_id) const;
    llm_pos seq_pos_max(llm_seq_id seq_id) const;

    uint32_t get_status()                                const { return status; }
    uint32_t get_n_kv(const slot_info & sinfo)           const;
    uint32_t get_n_kv()                         const noexcept { return n_kv; }
    void     set_n_kv(int n)                          noexcept { n_kv = n; }                 // deprecated
    uint32_t get_size() const;
    uint32_t get_n_para_kvcaches()                       const { return n_para_kvcaches; }
    size_t   get_i_cur()                                 const { return i_cur; }
    void     set_i_cur(size_t i)                      noexcept { i_cur = i; }
    size_t   size_k_bytes() const;
    size_t   size_v_bytes() const;

    void          build_input_k_idxs(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const llm_ubatch & ubatch);
    void          build_input_v_idxs(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const llm_ubatch & ubatch, llm_hparams& hparams);
    void          set_kq_mask(nnml_tensor * mask, int32_t buffer_id)     noexcept { self_kq_mask[buffer_id] = mask; }
    void          set_kq_mask_cnv(nnml_tensor * mask, int32_t buffer_id) noexcept { self_kq_mask_cnv[buffer_id] = mask; }
    nnml_tensor * get_k_idxs(int32_t buffer_id)                          const { return self_k_idxs[buffer_id]; }
    nnml_tensor * get_v_idxs(int32_t buffer_id)                          const { return self_v_idxs[buffer_id]; }
    nnml_tensor * get_kq_mask(int32_t buffer_id)                         const { return self_kq_mask_cnv[buffer_id]; }

    void set_inputs_attn_kidx(const llm_ubatch * ubatch);
    void set_inputs_attn_vidx(const llm_ubatch * ubatch, llm_hparams & hparams);
    void set_inputs_attn_mask(const llm_ubatch * ubatch);

    nnml_tensor * get_k(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, int32_t il, uint32_t n_kv, const slot_info & sinfo, llm_hparams & hparams);
    nnml_tensor * get_v(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, int32_t il, uint32_t n_kv, const slot_info & sinfo, llm_hparams & hparams);
    nnml_tensor * cpy_k(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * k_cur, nnml_tensor * k_idxs, int32_t il, const slot_info & sinfo, nnml_cgraph * cgraph);
    nnml_tensor * cpy_v(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * v_cur, nnml_tensor * v_idxs, int32_t il, const slot_info & sinfo, nnml_cgraph * cgraph);
    void clear(bool data);

private:
    int32_t    n_layer;
    int32_t    n_kv;
    uint32_t   n_para_kvcaches;
    uint32_t   n_pad;
    uint32_t   n_seq_max;
    uint32_t   n_stream;
    uint32_t   status = 0;
    size_t     i_cur;
    bool       v_trans = true;                                // indicate the value tensor is transposed
    
    struct kv_layer {
        uint32_t il;                                    // note: il is the layer index in the model, 
                                                        // which may be different from the layer index in the KV cache
        nnml_tensor * k[NNML_NUMA_MAX_NODES];
        nnml_tensor * v[NNML_NUMA_MAX_NODES];
        kv_layer() = default;
    };

    std::vector<kv_layer>                layers;
    nnml_tensor *                        self_k_idxs     [NNML_NUMA_MAX_NODES];
    nnml_tensor *                        self_v_idxs     [NNML_NUMA_MAX_NODES];
    nnml_tensor *                        self_kq_mask    [NNML_NUMA_MAX_NODES];
    nnml_tensor *                        self_kq_mask_cnv[NNML_NUMA_MAX_NODES];
    std::vector<uint32_t>                v_heads;
    std::vector<llm_kv_cells>            v_cells;
    std::vector<uint32_t>                seq_to_stream;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<slot_info>               sinfos;
    std::vector<llm_ubatch>              ubatches;
    // std::vector<llm_ubatch> ubatches;
};

using slot_info = llm_kv_cache::slot_info;


/**
 * nnml_tensor_ptrs: a wrapper class for a bundle of tensor pointers for parallel sub-graphs.
 *  - In parallel graph building, we need to maintain a bundle of tensor pointers for each tensor in the computation graph,
 *    because each sub-graph may have its own tensor instance for the same tensor.
 *  - There is no bound check on the index, because the usage is controlled in nnml_cgraph.
 */
class nnml_tensor_ptrs {
public:
    nnml_tensor_ptrs() noexcept {
        for (int i = 0; i < NNML_NUMA_MAX_NODES; ++i) group[i] = nullptr;
    }

    // we do not plan to check the index, because the index is either 1 or #buffers
    // the behavior is determined by the caller, e.g. nnml_cgraph
    nnml_tensor_ptrs(nnml_tensor * tensor) noexcept {
        group[0] = tensor;
        for (int i = 1; i < NNML_NUMA_MAX_NODES; ++i) group[i] = nullptr;
    }

    nnml_tensor_ptrs & operator=(nnml_tensor * tensor) noexcept {
        group[0] = tensor;
        return *this;
    } // we can use nnml_tensor_ptrs t = nnml_tensor * tensor;

    operator nnml_tensor *() const noexcept {
        return group[0];
    } // we can use nnml_tensor * tensor = nnml_tensor_ptrs t;

    nnml_tensor * & operator[](int idx) noexcept {
        return group[idx];
    } // we can use nnml_tensor * tensor = t[0]; and t[0] = tensor;

    nnml_tensor ** get_tensor_ptrs() noexcept {
        return group;
    }

    void set_name(const char * fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char name[NNML_MAX_NAME];
        vsnprintf(name, sizeof(name), fmt, args);
        va_end(args);
        for (int i = 0; i < NNML_NUMA_MAX_NODES; ++i) {
            if (group[i]) {
                group[i]->set_name("%s-N%d", name, i);
            }
        }
    }

private:
    nnml_tensor * group[NNML_NUMA_MAX_NODES];
};


/**
 * nnml_cgraph_cnode: a node in the sequential container.
 *  - Each node contains a tensor pointer to compute and the indices of its child nodes in the container.
 *  - This implementation is simple and straightforward for sub-graph parallelism.
 */
struct nnml_cgraph_cnode {
    nnml_tensor * tensor;
    int child[NNML_NUMA_MAX_NODES];

    nnml_cgraph_cnode() {
        tensor = nullptr;
        for (int i = 0; i < NNML_NUMA_MAX_NODES; ++i) child[i] = NNML_MAX_CGRAPH_CNODE;
    }
};


/**
 * nnml_cgraph: the main class for building computation graphs for LLM inference.
 *  - This class provides the interfaces for building the computation graph from the model definition, which is model-agnostic.
 *  - The actual graph building functions are implemented in the template functions, which can be registered for different models.
 *  - The graph is built in a sequential container of nnml_cgraph_cnode, which is simple and straightforward for sub-graph parallelism.
 *  - The sequential container is now implemented as a static list, which is simple but not robust enough.
 */
class nnml_cgraph {
public:
    nnml_cgraph() noexcept;
    nnml_cgraph(int64_t n_weight_tensors, llm_hparams &  hparams,   nnml_memory_t & mem,        bool is_tp,
                int     n_sub_graphs,     std::string    name,      bool            flash_attn, bool is_dual_buffer,
                bool    kv_unified,       llm_kv_cache * kv_cache,  bool is_eval = false) noexcept;
    ~nnml_cgraph() { graph_clear(); delete[] nodes; }

    // model-agnostic graph building interface
    template<typename model>
    void build_model_forward(model & mdl, bool is_tp = false) {
        auto it = registry.find(name);
        if (it != registry.end()) {
            it->second(*this, (void*)&mdl, is_tp);
        } else {
            NNML_ABORT("fatal error: graph builder not found for model %s", name.c_str());
        }
    }

    template<typename model>
    static void reg_builder(const std::string & key, void (*fn)(nnml_cgraph &, model &, bool)) {
        registry[key] = [fn](nnml_cgraph & graph, void * mdl_ptr, bool is_tp) {
            fn(graph, *static_cast<model *>(mdl_ptr), is_tp);
        };
    }

    // forward graph building functions
    nnml_tensor *       build_inp_embd(nnml_tensor * tok_embd);
    nnml_tensor_ptrs    build_inp_embd(nnml_tensor_ptrs tok_embd);
    nnml_tensor *       build_inp_pos(int i = 0);
    nnml_tensor_ptrs    build_inp_pos();
    void                build_attn_inp_kv();
    nnml_tensor *       build_inp_out_ids(int i = 0);
    nnml_tensor_ptrs    build_inp_out_ids();
    nnml_tensor *       build_norm(nnml_tensor * cur, nnml_tensor * mw, nnml_tensor * mb, llm_norm_type type, int il);
    nnml_tensor_ptrs    build_norm(nnml_tensor_ptrs cur, nnml_tensor ** mw, nnml_tensor ** mb, llm_norm_type type, int il);
    nnml_tensor *       build_lora_mm(nnml_tensor * w, nnml_tensor * cur);
    nnml_tensor_ptrs    build_lora_mm(nnml_tensor ** w, nnml_tensor_ptrs cur);
    nnml_tensor *       build_add(nnml_tensor * cur, nnml_tensor * w);
    nnml_tensor_ptrs    build_add(nnml_tensor_ptrs cur, nnml_tensor_ptrs w);
    nnml_tensor *       build_reshape_3d(nnml_tensor * cur, int64_t n_head, int64_t n_embd_head, int64_t n_tokens);
    nnml_tensor_ptrs    build_reshape_3d(nnml_tensor_ptrs cur, int64_t n_head, int64_t n_embd_head, int64_t n_tokens);
    nnml_tensor *       build_rope_ext(nnml_tensor * cur, nnml_tensor * b, nnml_tensor * c, int n_dims, int mode, int n_ctx_orig, float freq_base,
                                       float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
    nnml_tensor_ptrs    build_rope_ext(nnml_tensor_ptrs cur, nnml_tensor * b, nnml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base,
                                       float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
    nnml_tensor *       build_attn(nnml_tensor * wo, nnml_tensor * wo_b, nnml_tensor * q_cur, nnml_tensor * k_cur, nnml_tensor * v_cur,
                                   nnml_tensor * kq_b, nnml_tensor * sinks, nnml_tensor * v_mla, float kq_scale, int il);
    nnml_tensor_ptrs    build_attn(nnml_tensor ** wo, nnml_tensor ** wo_b, nnml_tensor_ptrs q_cur, nnml_tensor_ptrs k_cur, nnml_tensor_ptrs v_cur,
                                   nnml_tensor ** kq_b, nnml_tensor ** sinks, nnml_tensor ** v_mla, float kq_scale, int il);
    nnml_tensor *       build_get_rows(nnml_tensor * cur, nnml_tensor * ids);
    nnml_tensor_ptrs    build_get_rows(nnml_tensor_ptrs cur, nnml_tensor_ptrs ids);
    nnml_tensor *       build_ffn(nnml_tensor * cur, nnml_tensor * up, nnml_tensor * up_b, nnml_tensor * up_s,
                                  nnml_tensor * gate, nnml_tensor * gate_b, nnml_tensor * gate_s, nnml_tensor * down, nnml_tensor * down_b,
                                  nnml_tensor * down_s, nnml_tensor * act_scales, llm_ffn_op_type type_op, llm_ffn_gate_type type_gate, int il);
    nnml_tensor_ptrs    build_ffn(nnml_tensor_ptrs cur, nnml_tensor ** up, nnml_tensor ** up_b, nnml_tensor ** up_s,
                                  nnml_tensor ** gate, nnml_tensor ** gate_b, nnml_tensor ** gate_s, nnml_tensor ** down, nnml_tensor ** down_b,
                                  nnml_tensor ** down_s, nnml_tensor ** act_scales, llm_ffn_op_type type_op, llm_ffn_gate_type type_gate, int il);
    nnml_tensor_ptrs    build_scatter(nnml_tensor * src, bool is_by_row);
    nnml_tensor *       build_gather(nnml_tensor_ptrs src, bool is_by_row);

    void                build_forward_expand(nnml_tensor * tensor);
    void                build_forward_expand(nnml_tensor_ptrs tensor);
    void                graph_clear() {n_nodes = 0;}
    
    void                set_ubatch(const llm_ubatch & ubatch) noexcept { this->ubatch = ubatch; }
    void                set_inputs(const llm_ubatch * ubatch);
    // void                set_mem(nnml_memory_t & mem) noexcept { this->mem = std::move(mem); }           // for debug and test, std initializer is constructor only
    nnml_tensor *       get_logits() const noexcept { return t_logits; }
    nnml_tensor *       get_embd() const noexcept { return t_embd; }
    llm_kv_cache *      get_kvcache() const noexcept { return kvcache; }
    nnml_memory_t &     get_mem() noexcept { return mem; }
    llm_hparams &       get_hparams() noexcept { return hparams; }
    llm_ubatch &        get_ubatch() noexcept { return ubatch; }
    nnml_cgraph_cnode * get_nodes_ptr() const noexcept { return nodes; }
    nnml_cgraph_cnode * get_nth_cnode(int32_t idx) const noexcept { NNML_ASSERT(idx >= 0 && idx < n_nodes); return &nodes[idx]; }
    void                set_t_logits(nnml_tensor * t) noexcept { t_logits = t; }
    void                set_t_embd(nnml_tensor * t) noexcept { t_embd = t; }
    void                set_n_tokens(int64_t n) noexcept { n_tokens = n; }
    int64_t             get_n_tokens() const noexcept { return n_tokens; }
    int64_t             get_n_rot() const noexcept { return hparams.n_rot; }
    int64_t             get_n_head() const noexcept { return hparams.n_head_arr[0]; }
    int64_t             get_n_head_kv() const noexcept { return hparams.n_head_kv_arr[0]; }
    int32_t             get_n_kv_max() const noexcept { return n_kv_max; }
    int32_t             get_n_ctx_orig() const noexcept { return hparams.n_ctx_train; }
    int32_t             get_n_nodes() const noexcept { return n_nodes; }
    int32_t             get_n_para_graphs() const noexcept { return n_para_graphs; }
    int32_t             get_n_para_graphs_max() const noexcept { return n_para_graphs_max; }
    float               get_freq_base() const noexcept { return hparams.f_freq_base; }
    float               get_freq_scale() const noexcept { return hparams.f_freq_scale; }
    float               get_ext_factor() const noexcept { return hparams.f_ext_factor; }
    float               get_attn_factor() const noexcept { return hparams.f_attn_factor; }
    float               get_beta_fast() const noexcept { return hparams.f_beta_fast; }
    float               get_beta_slow() const noexcept { return hparams.f_beta_slow; }
    llm_rope_type       get_rope_type() const noexcept { return hparams.rope_type; }
    bool                is_flash_attn() const noexcept { return flash_attn; }
    bool                is_eval_mode() const noexcept { return is_eval; }

private:

    llm_ubatch          ubatch;
    llm_hparams         hparams;
    nnml_cgraph_cnode * nodes;              // legacy: nnml_tensor ** nodes;
    int64_t             n_tokens;
    int64_t             n_outputs;
    int32_t             n_nodes;
    int32_t             n_cnode_array;
    int32_t             n_para_graphs;      // this indicates how many parallel sub-graphs are using now (run-time), the real n_sub_graph is in llm_kv_cache
    int32_t             n_para_graphs_max;  // this is the maximum number of parallel sub-graphs, will be used to replace llm_kv_cache's n_para_kvcaches
    bool                is_para_mode;       // this indicates whether we are in parallel sub-graph computing now (run-time)
    bool                is_tp;              // this indicates whether tensor parallelism is enabled in the current inference
    bool                is_eval;
    // important tensor
    nnml_tensor *       t_tokens;
    nnml_tensor *       t_logits;
    nnml_tensor *       t_embd;
    nnml_tensor *       t_out_ids;
    nnml_tensor *       t_pos;
    // llm_lora_weight * loras;             // we do not support lora in the current version

    // kvcache setting
    llm_kv_cache *      kvcache;
    int32_t             n_kv_max;
    bool                kv_unified;
    bool                flash_attn;

    // buffer configuration
    nnml_memory_t       mem;
    bool                is_dual_buffer;
    // for model register
    std::string         name;
    static std::unordered_map<std::string, std::function<void(nnml_cgraph &, void *, bool)>> registry;

    nnml_tensor *    build_attn_mha(nnml_tensor * q, nnml_tensor * k, nnml_tensor * v, nnml_tensor * kq_b, nnml_tensor * kq_mask, nnml_tensor * sinks, nnml_tensor * v_mla, float kq_scale, int il);
    nnml_tensor_ptrs build_attn_mha(nnml_tensor_ptrs q, nnml_tensor_ptrs k, nnml_tensor_ptrs v, nnml_tensor ** kq_b, nnml_tensor ** kq_mask, nnml_tensor ** sinks, nnml_tensor ** v_mla, float kq_scale, int il);
    void             build_attn_inp_kv_impl(const llm_ubatch & ubatch);
    void             build_forward_impl(nnml_tensor * tensor, bool expand);
    void             build_forward_impl(nnml_tensor_ptrs tensor, bool expand);
    // int inf_tp_dims();
    void             set_inputs_embd(const llm_ubatch * ubatch);
    void             set_inputs_pos(const llm_ubatch * ubatch);
    void             set_inputs_out_ids(const llm_ubatch * ubatch);
    void             set_inputs_attn_kv(const llm_ubatch * ubatch);
};


// APIs
NNML_API int64_t nnml_time_us(void);


/**
 * llm_kv_cells: manages the state of key-value cache cells for multiple sequences.
 * ACKNOWLEDGEMENT: This module is almost transplanted from llama.cpp, which we will optimize it later!
 * Each cell can be used by multiple sequences, and the class tracks which sequences are using which cells, 
 * as well as any positional shifts applied to the cells.
 */
class llm_kv_cells {
public:
    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLM_MAX_SEQ; ++s) {
            seq_pos[s].clear();
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        shift.resize(n);
        seq.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : *used.begin();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : *used.rbegin() + 1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llm_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llm_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llm_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llm_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llm_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llm_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llm_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        if (seq[i].none()) {
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llm_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, pos[i]);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llm_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llm_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, pos[i]);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llm_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLM_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llm_pos seq_pos_min(llm_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLM_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].begin()->second > 0);

        return seq_pos[seq_id].begin()->first;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llm_pos seq_pos_max(llm_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLM_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].rbegin()->second > 0);

        return seq_pos[seq_id].rbegin()->first;
    }

    // note: call only if the cell is not empty
    llm_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    // note: call only if the cell is not empty
    llm_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llm_pos p0, llm_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llm_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used.insert(i);
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llm_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llm_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llm_pos> pos;

    std::vector<llm_pos> shift;

    using seq_set_t = std::bitset<LLM_MAX_SEQ>;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    std::map<llm_pos, int> seq_pos[LLM_MAX_SEQ];

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llm_seq_id s, llm_pos p) {
        auto it = seq_pos[s].find(p);
        assert(it != seq_pos[s].end());

        if (--it->second == 0) {
            seq_pos[s].erase(it);
        }
    }

    void seq_pos_inc(llm_seq_id s, llm_pos p) {
        seq_pos[s][p]++;
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLM_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, pos[i]);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLM_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, pos[i]);
            }
        }
    }
};
