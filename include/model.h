/**
 * @file model.h
 * @brief The model structure and related functions for loading and managing LLM models.
 * 
 * Provides:
 *  - Definition of the `llm_model` structure, which encapsulates all information about the model architecture, weight tensors, tokenizer.
 *  - Functions for loading model parameters from a GGUF file, and estimating activation and KV cache sizes.
 *  - Utility functions for slicing tensors.
 * 
 * Designed for loading models in a simple manner.
 */
#pragma once

#include <stdio.h>
#include <variant>

#include "tensor.h"
#include "cgraph.h"
#include "tokenizer.h"


#define LLM_INITIAL_CTX_LENGTH 512

#define LLM_LOG_PRINT(...) do { fprintf(stdout, __VA_ARGS__); } while(0)
#define LLM_LOG(is_print, ...) do { if (is_print) { fprintf(stdout, __VA_ARGS__); } } while(0)
#define LLM_ERROR(...) do { fprintf(stderr, __VA_ARGS__); exit(1); } while(0)


/** type of every hyperparameter item, for building map elsewhere */
enum llm_hparams_item {
    LLM_ARCHITECTURE,
    LLM_NAME,
    LLM_N_LAYER,
    LLM_CTX_LENGTH,
    LLM_EMB_LENGTH,
    LLM_FFN_LENGTH,
    LLM_ATTN_HEADCOUNT,
    LLM_ATTN_HEADCOUNT_KV,
    LLM_ROPE_FREQ_BASE,
    LLM_ATTN_LNORM_EPS,
    LLM_ATTN_KEY_LENGTH,
    LLM_ATTN_VALUE_LENGTH,
    LLM_TOKENIZER_MODEL,                 // there properties are implemented in tokenizer.h
    LLM_TOKENIZER_PRE_MODEL,
    LLM_TOKENIZER_TOKENS,
    LLM_TOKENIZER_TOKEN_TYPE,
    LLM_TOKENIZER_MERGES,
    LLM_TOKENIZER_EOS_ID,
    LLM_TOKENIZER_BOS_ID,
    LLM_TOKENIZER_UNK_ID,
    LLM_TOKENIZER_PAD_ID,
    LLM_TOKENIZER_ADD_BOS,
    LLM_TOKENIZER_CHAT_TMPL,
    LLM_VOCAB_TYPE,
    LLM_QUANT_VERSION
};

/** type of every weight item, for building map elsewhere */
enum llm_weight_item {
    LLM_TOKEN_EMBEDDINGS,
    LLM_OUTPUT_PROJ,
    LLM_OUTPUT_NORM,
    LLM_ATTENTION_NORM,
    LLM_ATTENTION_Q,
    LLM_ATTENTION_Q_NORM,
    LLM_ATTENTION_K,
    LLM_ATTENTION_K_NORM,
    LLM_ATTENTION_V,
    LLM_ATTENTION_O,
    LLM_FFN_GATE,
    LLM_FFN_UP,
    LLM_FFN_DOWN,
    LLM_FFN_NORM,
};

// weight slicing strategy
enum llm_weight_slice {
    LLM_WEIGHT_SLICE_NONE,
    LLM_WEIGHT_SLICE_ROW,
    LLM_WEIGHT_SLICE_COL,
};

extern std::map<llm_weight_item, llm_weight_slice> weight_slice;


/**
 * weight tensor type and container in one layer, with a simple interface for allocation and deallocation.
 */
enum layer_tensor_type {
    WQ, WK, WV, WO,
    ATTN_NORM, ATTN_Q_NORM, ATTN_K_NORM,
    FFN_GATE, FFN_DOWN, FFN_UP, FFN_NORM,
    LAYER_TENSOR_COUNT
};

struct llm_layer {
    nnml_tensor ** tensors[LAYER_TENSOR_COUNT]{};

    void alloc_all(size_t n_nodes) {
        for (int i = 0; i < LAYER_TENSOR_COUNT; i++) {
            tensors[i] = new nnml_tensor*[n_nodes];
            for (size_t j = 0; j < n_nodes; j++) {
                tensors[i][j] = nullptr;
            }
        }
    }

    void free_all() {                                   // TODO: this func is not well used, check and add
        for (int i = 0; i < LAYER_TENSOR_COUNT; i++) {
            if (!tensors[i]) continue;
            delete[] tensors[i];
            tensors[i] = nullptr;
        }
    }
};

// tensor metadata loaded from the model file, used for loading weights
struct llm_tensor_meta {
    std::string name;
    uint32_t    n_dims;
    uint64_t    ne[4];
    uint32_t    ttype;
    uint64_t    offset;
    size_t      nbytes;
};


/**
 * llm_model: the main structure in a loaded LLM model, containing all necessary information and data for inference.
 */
struct llm_model {
    
    FILE *   model_file_ptr = nullptr;
    int32_t  n_tensor_nodes = 1;                         // number of NUMA nodes for tensor storage
    size_t   total_bytes  = 0;
    size_t   act_layer_bytes = 0;
    size_t   act_total_bytes = 0;
    size_t   kv_cache_bytes_token = 0;
    size_t   kv_cache_bytes = 0;
    uint64_t total_params = 0;
    uint64_t n_tensors = -1;
    uint64_t n_kv = -1;
    bool     is_asm_gemm = false;

    std::string name = "n/a";
    std::string chat_template;
    
    llm_tensor_meta * tensor_meta = nullptr;
    llm_hparams       hparams;
    llm_vocab         tokenizer;
    vocab_data        tokenizer_data;

    nnml_cgraph * cgraph = nullptr;                 // maybe deprecated, cgraph should not be tightly coupled with the model
    nnml_tensor * tok_embd        = nullptr;
    nnml_tensor * output_norm     = nullptr;
    nnml_tensor * output          = nullptr;

    std::vector<llm_layer> layers;

    void load_gguf_kv(bool is_print);
    // void load_vocab();               // maybe deprecated, since vocab is loaded in tokenizer
    bool load_all_tensors(std::map<std::string, llm_weight_item> & weight_map, nnml_memory_t& mem);
    void set_asm_gemm(nnml_tensor ** tensors);

    size_t get_n_tensors() const;
    // const nnml_tensor * get_tensor(const char * name) const;         // for debug in the future
    // void print_info() const;                                         // for debug in the future

    void open_model_file(const char * filename) { assert(model_file_ptr == nullptr); model_file_ptr = fopen(filename, "rb"); }
    void close_model_file() { if (model_file_ptr != nullptr) fclose(model_file_ptr); model_file_ptr = nullptr; }
    void compute_params();
    void esti_activation(uint32_t batch);
    void esti_kvcache(uint32_t ctx_len);
};

void slice_tensor(nnml_memory_t& mem, nnml_tensor * src, size_t src_nbytes, nnml_tensor **& dst_arr, llm_weight_slice slice_type, int n_tensor_nodes, int n_attn_hdim);
