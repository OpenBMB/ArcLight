/**
 * @file model.cpp
 * @brief Implementation of model.h
 * 
 * This file implements the functions declared in model.h for loading and managing LLM models,
 * including loading parameters from a GGUF file, estimating activation and KV cache sizes, and slicing tensors.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "cgraph.h"
#include "ops.h"
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "models.h"

std::map<llm_weight_item, llm_weight_slice> weight_slice = {            // weight slicing strategy, need to be checked
    {LLM_ATTENTION_Q, LLM_WEIGHT_SLICE_ROW},
    {LLM_ATTENTION_K, LLM_WEIGHT_SLICE_ROW},
    {LLM_ATTENTION_V, LLM_WEIGHT_SLICE_ROW},
    {LLM_ATTENTION_O, LLM_WEIGHT_SLICE_COL},
    {LLM_FFN_UP,      LLM_WEIGHT_SLICE_ROW},
    {LLM_FFN_GATE,    LLM_WEIGHT_SLICE_ROW},
    {LLM_FFN_DOWN,    LLM_WEIGHT_SLICE_COL},
};

static uint32_t gcd(uint32_t a, uint32_t b) {
    while (b) { a %= b; std::swap(a, b); }
    return a;
}
static uint32_t lcm(uint32_t a, uint32_t b) {
    if (a == 0 || b == 0) return 0;
    return (a * b) / gcd(a, b);
}


/**
 * tensor slicing function, used for slicing large weight tensors into smaller ones
 * according to the specified slicing strategy, to achieve better tensor parallelism across NUMA nodes.
 */
void slice_tensor(nnml_memory_t& mem, nnml_tensor * src, size_t src_nbytes, nnml_tensor **& dst_arr, llm_weight_slice slice_type, int n_tensor_nodes, int n_attn_hdim) {
    assert(src != nullptr);
    assert(dst_arr != nullptr);
    assert(src->n_dims() == 2);
    if (n_tensor_nodes > 1 && (slice_type == LLM_WEIGHT_SLICE_ROW || slice_type == LLM_WEIGHT_SLICE_COL)) {
        nnml_type type = (nnml_type)src->get_data_type();
        uint32_t bsize = nnml_blck_size(type);
        // calculating a uniform grain size
        // the split point must satisfy both hardware/data alignment (bsize) and logical alignment (n_attn_hdim)
        uint32_t grain_size = (n_attn_hdim > 0) ? lcm(bsize, (uint32_t)n_attn_hdim) : bsize;
        // determine which dimension to cut
        uint32_t dim_to_slice = (slice_type == LLM_WEIGHT_SLICE_ROW) ? 1 : 0;
        uint32_t total_elements = (uint32_t)src->get_elements(dim_to_slice);
        // calculate the total number of alignment units
        assert(total_elements % grain_size == 0 && "Total elements must be multiple of grain_size");
        uint32_t total_units = total_elements / grain_size;
        uint32_t units_per_node = total_units / n_tensor_nodes;
        uint32_t units_remainder = total_units % n_tensor_nodes;
        uint32_t slice_elements[NNML_NUMA_MAX_NODES] = {0};
        size_t start_offset_elements[NNML_NUMA_MAX_NODES] = {0};
        for (int i = 0; i < n_tensor_nodes; ++i) {
            slice_elements[i] = (units_per_node + (i < units_remainder ? 1 : 0)) * grain_size;
        }
        for (int i = 1; i < n_tensor_nodes; ++i) {
            start_offset_elements[i] = start_offset_elements[i-1] + slice_elements[i-1];
        }

        // assert(slice_type == LLM_WEIGHT_SLICE_ROW || slice_type == LLM_WEIGHT_SLICE_COL);
        if (slice_type == LLM_WEIGHT_SLICE_ROW) {
            size_t nbytes_per_row = src->get_stride_bytes(1);
            for (int i = 0; i < n_tensor_nodes; ++i) {
                size_t nbytes_node = nbytes_per_row * slice_elements[i];
                void * const obj_new = mem->allocate_obj(NNML_TENSOR_TYPE_WEIGHT, i, 0, NNML_TENSOR_SIZE + nbytes_node);
                memcpy(obj_new, src, NNML_TENSOR_SIZE);
                nnml_tensor * dst = (nnml_tensor *)obj_new;
                uint8_t * dst_data = (uint8_t *)obj_new + NNML_TENSOR_SIZE;
                dst->set_tensor_data(dst_data);
                memcpy(dst_data, (uint8_t*)src->tensor_data() + start_offset_elements[i] * nbytes_per_row, nbytes_node);
                dst->set_elements(1, slice_elements[i]);
                // we assume that the weights to be sliced are always 2-D
                dst_arr[i] = dst;
            }
        } else {
            uint32_t total_rows = (uint32_t)src->get_elements(1);
            size_t type_size = nnml_type_size(type);
            for (int i = 0; i < n_tensor_nodes; ++i) {
                // when splitting columns, the number of bytes occupied by each row changes
                size_t nbytes_per_row_dst = (slice_elements[i] / bsize) * type_size;
                size_t nbytes_node = nbytes_per_row_dst * total_rows;
                void * const obj_new = mem->allocate_obj(NNML_TENSOR_TYPE_WEIGHT, i, 0, NNML_TENSOR_SIZE + nbytes_node);
                memcpy(obj_new, src, NNML_TENSOR_SIZE);
                nnml_tensor * dst = (nnml_tensor *)obj_new;
                uint8_t * dst_data = (uint8_t *)obj_new + NNML_TENSOR_SIZE;
                dst->set_tensor_data(dst_data);
                // copy row by row
                for (int j = 0; j < (int)total_rows; ++j) {
                    uint8_t * row_src = (uint8_t *)src->tensor_data() + j * src->get_stride_bytes(1) + 
                                    (start_offset_elements[i] / bsize) * type_size;
                    uint8_t * row_dst = dst_data + j * nbytes_per_row_dst;
                    memcpy(row_dst, row_src, nbytes_per_row_dst);
                }
                dst->set_elements(0, slice_elements[i]);
                dst->set_stride_bytes(1, nbytes_per_row_dst);
                dst_arr[i] = dst;
            }
        }
        free(src);
    } else if (n_tensor_nodes > 1 && slice_type == LLM_WEIGHT_SLICE_NONE) {
        for (int i = 0; i < n_tensor_nodes; ++i) {
            void * const obj_new = mem->allocate_obj(NNML_TENSOR_TYPE_WEIGHT, i, 0, NNML_TENSOR_SIZE + src_nbytes);
            memcpy(obj_new, src, NNML_TENSOR_SIZE + src_nbytes);
            ((nnml_tensor *)obj_new)->set_tensor_data((void *)((uint8_t *)obj_new + NNML_TENSOR_SIZE));
            dst_arr[i] = (nnml_tensor *)obj_new;
        }
        free(src);
    } else if (n_tensor_nodes == 1) {
        void * const obj_new = mem->allocate_obj(NNML_TENSOR_TYPE_WEIGHT, 0, 0, NNML_TENSOR_SIZE + src_nbytes);
        memcpy(obj_new, src, NNML_TENSOR_SIZE + src_nbytes);
        ((nnml_tensor *)obj_new)->set_tensor_data((void *)((uint8_t *)obj_new + NNML_TENSOR_SIZE));
        
        dst_arr[0] = (nnml_tensor *)obj_new;
        free(src);
    } else {
        LLM_ERROR("invalid n_tensor_nodes %d", n_tensor_nodes);
    }
}

// implementation of llm_model
void llm_model::load_gguf_kv(bool is_print) {
    assert(model_file_ptr != nullptr);
    FILE * f = model_file_ptr;

    char magic[4];
    TRYREAD(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) LLM_ERROR("not a GGUF file");

    uint32_t version   = r_u32(f);
    uint64_t n_tensors = r_u64(f);
    uint64_t n_kv      = r_u64(f);
    LLM_LOG(true, "Magic: GGUF\nVersion: %u\nKV count: %" PRIu64 "\nTensor count: %" PRIu64 "\n\n",
           version, n_kv, n_tensors);
    this->n_tensors = n_tensors;
    this->n_kv = n_kv;

    uint64_t alignment = 32;            // default
    char *key = r_str(f);               // read the model architecture key
    uint32_t vtype = r_u32(f);
    LLM_LOG(is_print, "[%4" PRIu64 "] %s = ", 0, key);
    char *s = r_str(f);
    LLM_LOG(is_print, "\"%.*s\"\n", 100, s);
    name = std::string(s);              // we use the architecture as the model name to find the corresponding hparams map
    free(s);
    free(key);
    std::map<std::string, llm_hparams_item> & param_map = get_model_hparams_map(name);
    llm_hparams_item value;
    for (uint64_t i = 1; i < n_kv; ++i){
        char *key = r_str(f);
        uint32_t vtype = r_u32(f);
        assert(key != nullptr);
        LLM_LOG(is_print, "[%4" PRIu64 "] %s = ", i, key);
        auto it = param_map.find(std::string(key));
        if (key && it != param_map.end()) {
            value = it->second;
            if (vtype == GGUF_TYPE_ARRAY){
                // print_array(f);                     // this function will be modified to load arrays such as tokenizer
                switch (value) {
                    case LLM_TOKENIZER_TOKENS: {
                        std::vector<std::string> tokens;
                        read_array_str(f, tokens);
                        tokenizer_data.vocab.resize(tokens.size());
                        for (size_t j = 0; j < tokens.size(); ++j) {
                            tokenizer_data.vocab[j].text = tokens[j];
                        }
                        LLM_LOG(is_print, "tokens...\n");
                        break;
                    }
                    case LLM_TOKENIZER_TOKEN_TYPE: {
                        tokenizer_data.has_type = true;
                        std::vector<int32_t> token_types;
                        read_array_i32(f, token_types);
                        assert(token_types.size() == tokenizer_data.vocab.size());
                        for (size_t j = 0; j < token_types.size(); ++j) {
                            tokenizer_data.vocab[j].type = (llm_token_type)token_types[j];
                        }
                        LLM_LOG(is_print, "token types...\n");
                        break;
                    }
                    case LLM_TOKENIZER_MERGES: {
                        std::vector<std::string> merges;
                        read_array_str(f, merges);
                        tokenizer_data.merges.resize(merges.size());
                        for (size_t j = 0; j < merges.size(); ++j) {
                            tokenizer_data.merges[j] = merges[j];
                        }
                        LLM_LOG(is_print, "merges array...\n");
                        break;
                    }
                    default:
                        print_array(f, is_print);
                        break;
                }
            } else if (vtype == GGUF_TYPE_STRING){
                char *s = r_str(f);
                switch (value) {
                    case LLM_ARCHITECTURE:
                        // load_arch(s);
                        break;
                    case LLM_NAME:
                        // name = std::string(s);
                        break;
                    case LLM_TOKENIZER_CHAT_TMPL:
                        chat_template = std::string(s);
                        break;
                    case LLM_TOKENIZER_MODEL:
                        tokenizer_data.model = std::string(s);
                        break;
                    case LLM_TOKENIZER_PRE_MODEL:
                        // handle pre model string if needed
                        tokenizer_data.pre_model = std::string(s);
                        break;
                    default:
                        break;
                }
                if (strlen(s) >= 10) s[10] = '\0';
                LLM_LOG(is_print, "\"%.*s\"\n", 100, s);
                free(s);
            } else {
                switch (value) {
                    case LLM_N_LAYER:
                        hparams.n_layer = r_u32(f);
                        layers.resize(hparams.n_layer);
                        for(int il = 0; il < hparams.n_layer; il++) layers[il].alloc_all(n_tensor_nodes);
                        LLM_LOG(is_print, "%u\n", hparams.n_layer);
                        break;
                    case LLM_CTX_LENGTH:
                        hparams.n_ctx_train = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_ctx_train);
                        break;
                    case LLM_EMB_LENGTH:
                        hparams.n_embd = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_embd);
                        break;
                    case LLM_FFN_LENGTH:
                        hparams.n_intermediate_size = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_intermediate_size);
                        break;
                    case LLM_ATTN_HEADCOUNT:
                        hparams.n_head_arr[0] = r_u32(f);
                        for(int il = 1; il < hparams.n_layer; il++) hparams.n_head_arr[il] = hparams.n_head_arr[0];
                        LLM_LOG(is_print, "%u\n", hparams.n_head_arr[0]);
                        hparams.n_embd_head_k = hparams.n_embd_head_v = hparams.n_rot = hparams.n_embd / hparams.n_head();
                        break;
                    case LLM_ATTN_HEADCOUNT_KV:
                        hparams.n_head_kv_arr[0] = r_u32(f);
                        for(int il = 1; il < hparams.n_layer; il++) hparams.n_head_kv_arr[il] = hparams.n_head_kv_arr[0];
                        LLM_LOG(is_print, "%u\n", hparams.n_head_kv_arr[0]);
                        break;
                    case LLM_ROPE_FREQ_BASE:
                        hparams.f_rope_freq_base_train = r_f32(f);
                        LLM_LOG(is_print, "%g\n", hparams.f_rope_freq_base_train);
                        break;
                    case LLM_ATTN_LNORM_EPS:
                        hparams.f_norm_rms_eps = r_f32(f);
                        LLM_LOG(is_print, "%g\n", hparams.f_norm_rms_eps);
                        break;
                    case LLM_ATTN_KEY_LENGTH:
                        hparams.n_embd_head_k = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_embd_head_k);
                        hparams.n_rot = hparams.n_embd_head_k;
                        break;
                    case LLM_ATTN_VALUE_LENGTH:
                        hparams.n_embd_head_v = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_embd_head_v);
                        break;
                    case LLM_TOKENIZER_EOS_ID:
                        tokenizer_data.eos_id = r_u32(f);
                        LLM_LOG(is_print, "%u\n", tokenizer_data.eos_id);
                        break;
                    case LLM_TOKENIZER_BOS_ID:
                        tokenizer_data.bos_id = r_u32(f);
                        LLM_LOG(is_print, "%u\n", tokenizer_data.bos_id);
                        break;
                    case LLM_TOKENIZER_PAD_ID:
                        tokenizer_data.pad_id = r_u32(f);
                        LLM_LOG(is_print, "%u\n", tokenizer_data.pad_id);
                        break;
                    case LLM_TOKENIZER_ADD_BOS:
                        tokenizer_data.add_bos = (bool)r_u8(f);
                        tokenizer_data.read_add_bos = true;
                        LLM_LOG(is_print, "%u\n", tokenizer_data.add_bos);
                        break;
                    default:
                        // consume but ignore other known params for now
                        print_scalar(f, vtype, is_print);
                        break;
                }
            }
        } else {
            // unknown param, just print
            if (vtype == GGUF_TYPE_ARRAY){
                print_array(f, is_print);
            } else {
                print_scalar(f, vtype, is_print);
            }
        }
        free(key);
    }
    
    hparams.n_rot = hparams.n_embd_head_k;

    tensor_meta = new llm_tensor_meta[n_tensors];   // just use new here for simplicity
                                                    // it should be free when the application exits
    for (uint64_t i = 0; i < n_tensors; ++i) {
        char *name  = r_str(f);
        tensor_meta[i].name = std::string(name);
        uint32_t nd = r_u32(f);
        tensor_meta[i].n_dims = nd;
        if (nd > 4) LLM_ERROR("tensor '%s' has too many dims: %u", name, nd);

        uint64_t ne[4] = {1,1,1,1};
        for (uint32_t d = 0; d < nd; ++d) ne[d] = r_u64(f);
        memcpy(tensor_meta[i].ne, ne, sizeof(ne));

        uint32_t ttype  = r_u32(f);
        uint64_t offset = r_u64(f);
        tensor_meta[i].ttype = ttype;
        tensor_meta[i].offset = offset;
        free(name);
    }
    uint64_t offset_base = GGUF_ALIGN_UP(ftell(f));
    for (uint64_t i = 0; i < n_tensors; ++i) tensor_meta[i].offset += offset_base;

    compute_params();
    esti_activation(LLM_INITIAL_CTX_LENGTH);
    esti_kvcache(LLM_INITIAL_CTX_LENGTH);
}

void llm_model::set_asm_gemm(nnml_tensor ** tensors) {
    for (int i = 0; i < n_tensor_nodes; ++i) {
        nnml_tensor * weight_tensor = tensors[i];
        if (weight_tensor == nullptr) continue;
        weight_tensor->set_asm_gemm();
        // printf("repack...\n");
        int repack_ret = format_repack(weight_tensor);
        if (repack_ret != 0) {
            LLM_ERROR("tensor '%s' repack failed", weight_tensor->get_name_ptr());
        }
    }
}

bool llm_model::load_all_tensors(std::map<std::string, llm_weight_item> & weight_map, nnml_memory_t& mem) {
    assert(model_file_ptr != nullptr);
    FILE * f = model_file_ptr;
    int n_attn_hdim = hparams.n_embd_head_k;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        if (i % 2 == 0 || i == n_tensors - 1) {
            double progress = (double)(i + 1) / n_tensors;
            int bar_width = 100;
            int pos = (int)(bar_width * progress);

            LLM_LOG_PRINT("\r[");
            for (int j = 0; j < bar_width; ++j)
                putchar(j < pos ? '=' : (j == pos ? '>' : ' '));
            LLM_LOG_PRINT("] %5.1f%%", progress * 100);
            fflush(stdout);
        }

        void * const obj_new = malloc(NNML_TENSOR_SIZE + tensor_meta[i].nbytes);
        nnml_tensor * weight_tensor = (nnml_tensor *)obj_new;
        weight_tensor->set_data_type((nnml_type)tensor_meta[i].ttype);
        weight_tensor->set_tensor_type(NNML_TENSOR_TYPE_WEIGHT);
        weight_tensor->set_operation(NNML_OP_NONE);
        strcpy(weight_tensor->get_name_ptr(), tensor_meta[i].name.c_str());
        memcpy(weight_tensor->get_ne_ptr(), tensor_meta[i].ne, sizeof(tensor_meta[i].ne));
        size_t nb[4] = {1, 1, 1, 1};
        nb[0] = nnml_type_size((nnml_type)tensor_meta[i].ttype);
        if (nnml_blck_size((nnml_type)tensor_meta[i].ttype) > 1) {
            nb[1] = nb[0] * (tensor_meta[i].ne[0] / nnml_blck_size((nnml_type)tensor_meta[i].ttype));
            for (int j = 2; j < tensor_meta[i].n_dims; ++j) nb[j] = nb[j-1] * tensor_meta[i].ne[j-1];
        } else {
            for (int j = 1; j < tensor_meta[i].n_dims; ++j) nb[j] = nb[j-1] * tensor_meta[i].ne[j-1];
        }
        memcpy(weight_tensor->get_nb_ptr(), nb, sizeof(nb));
        weight_tensor->set_tensor_data((void *)((uint8_t *)obj_new + NNML_TENSOR_SIZE));

        fseek(f, (long)tensor_meta[i].offset, SEEK_SET);
        size_t tmp = fread(weight_tensor->tensor_data(), tensor_meta[i].nbytes, 1, f);
        
        std::string tensor_type_str;
        int layer_idx = -1;
        std::string tensor_name = std::string(tensor_meta[i].name.c_str());
        std::size_t pos = tensor_name.find_first_of("0123456789");
        if (pos == std::string::npos) {
            tensor_type_str = tensor_name;
        } else {
            tensor_type_str = tensor_name.substr(0, pos);
            std::size_t end = pos;
            while (end < tensor_name.size() && std::isdigit(tensor_name[end])) ++end;
            layer_idx = std::stoi(tensor_name.substr(pos, end - pos));
            if (end < tensor_name.size()) {
                if (tensor_name[end] == '.') ++end;
                tensor_type_str = tensor_name.substr(end);
            }
        }
        for (auto &pair : weight_map) {
            const std::string &key = std::string(pair.first);
            if (tensor_type_str.find(key) != std::string::npos) {
                llm_weight_item item = pair.second;
                switch (item) {
                    case LLM_TOKEN_EMBEDDINGS:
                        tok_embd = weight_tensor;
                        break;
                    case LLM_ATTENTION_Q:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[WQ], weight_slice[item], n_tensor_nodes, n_attn_hdim);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[WQ]);
                        break;
                    case LLM_ATTENTION_K:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[WK], weight_slice[item], n_tensor_nodes, n_attn_hdim);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[WK]);
                        break;
                    case LLM_ATTENTION_V:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[WV], weight_slice[item], n_tensor_nodes, n_attn_hdim);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[WV]);
                        break;
                    case LLM_ATTENTION_O:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[WO], weight_slice[item], n_tensor_nodes, n_attn_hdim);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[WO]);
                        break;
                    case LLM_ATTENTION_NORM:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[ATTN_NORM], LLM_WEIGHT_SLICE_NONE, 1, 0);
                        break;
                    case LLM_ATTENTION_Q_NORM:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[ATTN_Q_NORM], LLM_WEIGHT_SLICE_NONE, n_tensor_nodes, 0);
                        break;
                    case LLM_ATTENTION_K_NORM:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[ATTN_K_NORM], LLM_WEIGHT_SLICE_NONE, n_tensor_nodes, 0);
                        break;
                    case LLM_FFN_GATE:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[FFN_GATE], weight_slice[item], n_tensor_nodes, 0);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[FFN_GATE]);
                        break;
                    case LLM_FFN_UP:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[FFN_UP], weight_slice[item], n_tensor_nodes, 0);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[FFN_UP]);
                        break;
                    case LLM_FFN_DOWN:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[FFN_DOWN], weight_slice[item], n_tensor_nodes, 0);
                        if (is_asm_gemm) set_asm_gemm(layers[layer_idx].tensors[FFN_DOWN]);
                        break;
                    case LLM_FFN_NORM:
                        slice_tensor(mem, weight_tensor, tensor_meta[i].nbytes, layers[layer_idx].tensors[FFN_NORM], LLM_WEIGHT_SLICE_NONE, 1, 0);
                        break;
                    case LLM_OUTPUT_NORM:
                        output_norm = weight_tensor;
                        break;
                    case LLM_OUTPUT_PROJ:
                        output = weight_tensor;
                        break;
                    default:
                        LLM_ERROR("unhandled weight item %d for tensor %s\n", item, tensor_meta[i].name.c_str());
                        break;
                }
                break;
            }
        }
    }
    LLM_LOG(true, "\nAll %ld tensors loaded successfully.\n", n_tensors);
    return true;
}

void llm_model::compute_params() {
    double total_bytes = 0.0;
    double total_params = 0.0;
    for (size_t i = 0; i < n_tensors; ++i) {
        uint64_t ne_total = 1;
        for (uint32_t d = 0; d < tensor_meta[i].n_dims; ++d)
            ne_total *= tensor_meta[i].ne[d];
        tensor_meta[i].nbytes = (ne_total / nnml_blck_size((nnml_type)tensor_meta[i].ttype)) * nnml_type_size((nnml_type)tensor_meta[i].ttype);
        total_bytes += tensor_meta[i].nbytes;
        total_params += ne_total;
    }
    this->total_bytes = (size_t)total_bytes;
    this->total_params = (uint64_t)total_params;
}

void llm_model::esti_activation(uint32_t batch) {
    size_t bytes_per_token = (size_t)(hparams.n_embd * 10 * sizeof(float)); // per layer per token
    act_layer_bytes = (size_t) bytes_per_token * hparams.n_layer;
    act_total_bytes = act_layer_bytes * batch;
}

void llm_model::esti_kvcache(uint32_t ctx_len) {
    size_t bytes_per_entry_k = hparams.n_embd_head_k * sizeof(float);
    kv_cache_bytes_token = 2.0 * hparams.n_layer * hparams.n_head_arr[0] * bytes_per_entry_k;
    kv_cache_bytes = kv_cache_bytes_token * ctx_len;
}

size_t llm_model::get_n_tensors() const {
    return n_tensors;
}
