/**
 * @file context.h
 * @brief decoding context management for LLM inference.
 * 
 * Provides:
 *  - Decoding logic encapsulated in the `decoding_context` struct,
 *    which manages the computation graph (from nnml), scheduler (from nnml), and sampling.
 */
#pragma once

#include <array>
#include <vector>
#include <set>
#include <memory>
#include <unordered_map>

#include "cgraph.h"
#include "model.h"
#include "scheduler.h"
#include "sampler.h"


/**
 * a helper struct for managing the decoding params
 * this current implementation need to be further refactored and optimized, but it serves the purpose for now
 */
struct llm_cparams {
    llm_cparams(uint32_t n_ctx_ = 1024, uint32_t n_batch_ = 512, uint32_t n_ubatch_ = 256, uint32_t n_seq_max_ = 2,
                int32_t n_threads_ = 4, float rope_freq_base_ = 10000.0f, float rope_freq_scale_ = 1.0f,
                bool flash_attn_ = false, bool no_perf_ = false, bool kv_unified_ = false)
        : n_ctx(n_ctx_), n_batch(n_batch_), n_ubatch(n_ubatch_), n_seq_max(n_seq_max_),
          n_threads(n_threads_), rope_freq_base(rope_freq_base_), rope_freq_scale(rope_freq_scale_),
          flash_attn(flash_attn_), no_perf(no_perf_), kv_unified(kv_unified_) {}
    
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    float    rope_freq_base;
    float    rope_freq_scale;
    bool     flash_attn;
    bool     no_perf;
    bool     kv_unified;
};


/**
 * a helper struct for adstracting the batch in decoding
 */
struct llm_batch {
    int32_t         n_tokens;
    llm_token    *  token;
    float        *  embd;
    llm_pos      *  pos;
    int32_t      *  n_seq_id;
    llm_seq_id  **  seq_id;
    int8_t       *  logits;
};

llm_batch llm_batch_get_one(llm_token * tokens, int32_t n_tokens);

class llm_batch_allocr;


/**
 * decoding context: core controller for LLM inference.
 */
struct decoding_context {

    bool is_tp  = false;

    int32_t    n_reused = 0;
    uint32_t   n_stream = 1;
    int64_t    t_prefill_us = 0;
    uint32_t   n_prefill_tokens = 0;
    int64_t    t_decode_us = 0;
    uint32_t   n_decode_tokens = 0;
    
    llm_cparams        cparams;
    nnml_tensor      * res_logits = nullptr;
    nnml_scheduler   * scheduler = nullptr;
    nnml_cgraph      * cgraph = nullptr;
    llm_model        * model  = nullptr;
    llm_batch_allocr * balloc = nullptr;
    sampler_type       samp_type = SAMPLER_TYPE_NONE;
    sampler_params     samp_params;

    std::vector<int32_t> output_ids;
    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };
    std::vector<swap_info> output_swaps;    // for multi-token, is not tested yet

    void reset() { n_reused = 0;}
    void clear();
    bool init_batch(const llm_batch & batch_inp, uint32_t n_embd, uint32_t n_seq_max, bool output_all);
    bool init_ubatch(uint32_t n_ubatch);
    llm_ubatch get_ubatch();

    int  decode(const llm_batch & batch_inp);
    bool process_ubatch(const llm_ubatch & ubatch);
    void infer(std::vector<llm_token> & input_tokens, int max_new_tokens);
    // void batch_infer(const std::vector<llm_token> & input_tokens, int max_new_tokens);       // maybe deprecated

    void print_perf() const;

    std::function<bool()> token_callback = nullptr;
};


// a helper for sanitizing, fulfilling and splitting a batch
// this class is borrowed from llama.cpp
// it may be optimized in the future
class llm_batch_allocr {
public:
    llm_batch_allocr(uint32_t n_pos_per_embd);

    // sanitize and auto-gen missing data in the input batch
    // memory is optional. if provided will be used to check for sequence continuity and to determine the positions
    bool init(
            const llm_batch & batch_inp,
            const llm_vocab & vocab,
            const llm_kv_cache * kvcache,
            uint32_t n_embd,
            uint32_t n_seq_max,
            bool output_all);

    const llm_batch & get_batch() const;

    uint32_t get_n_tokens()  const;
    uint32_t get_n_outputs() const;
    uint32_t get_n_used()    const;

    // the array of output indices in the order they were encountered during the ubatch splitting
    std::vector<int32_t> & get_out_ids();

    // min/max positions of each sequence in the current ubatch
    llm_pos seq_pos_min(llm_seq_id seq_id) const;
    llm_pos seq_pos_max(llm_seq_id seq_id) const;

    // call once before splitting the batch to reset the internal state
    void split_reset();

    // simple split, unknown number of sequence sets of unequal lengths
    llm_ubatch split_simple(uint32_t n_ubatch);

    // make ubatches of equal-length sequences sets
    // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
    llm_ubatch split_equal(uint32_t n_ubatch, bool sequential);

    // sequence-set-wise split - each ubatch contains a single sequence-set
    llm_ubatch split_seq(uint32_t n_ubatch);

    // a helper method for creating a well-defined ubatch of tokens
    llm_ubatch ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs);

private:
    void clear();

    // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
    // return llama_ubatch.n_tokens == 0 if the entire batch was consumed
    llm_ubatch ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs);

    // for debugging, start with LLAMA_BATCH_DEBUG=2
    void ubatch_print(const llm_ubatch & ubatch, int debug);

    llm_batch batch;

    // only for debugging purposes
    const llm_vocab * vocab;

    const uint32_t n_pos_per_embd;

    uint32_t n_embd;
    uint32_t n_seq_max;
    uint32_t n_outputs;

    std::array<llm_seq_id, 1> seq_id_0 = { 0 }; // default sequence id

    std::vector<llm_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llm_seq_id *> seq_id;
    std::vector<llm_seq_id>   seq_id_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;

    using pos_set_t = std::set<llm_pos>;
    using seq_cpl_t = std::vector<bool>;

    // helper flag to quickly determine if there are any coupled sequences in the batch
    bool has_cpl = false;

    std::vector<pos_set_t> seq_pos; // seq_pos[s]: the set of positions in sequence s
    std::vector<seq_cpl_t> seq_cpl; // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

    using idx_vec_t = std::vector<int32_t>;
    using seq_set_t = std::bitset<LLM_MAX_SEQ>;

    std::vector<seq_set_t> seq_set; // seq_set[i]: the sequence set of token i

    std::unordered_map<seq_set_t, idx_vec_t> seq_set_map; // the indices at which the sequence set appears

    // batch indices of the output
    std::vector<int32_t> out_ids;

    uint32_t n_used;

    // used[i] indicates if token i has already been used in a previous ubatch
    std::vector<bool> used;

    int debug;
};