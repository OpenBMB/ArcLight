/**
 * @file gguf.cpp
 * @brief Implementation of gguf.h
 * 
 * This file implements the functions declared in gguf.h for reading and printing scalar and array values from a GGUF file.
 */
#include <vector>
#include <string>
#include "gguf.h"


void read_array_str(FILE *f, std::vector<std::string> & out_vec) {
    uint32_t elem_t = r_u32(f);
    uint64_t n = r_u64(f);
    if (elem_t != GGUF_TYPE_STRING)
        LLM_ERROR("expected string array element type but got %u\n", elem_t);
    for (uint64_t i = 0; i < n; ++i) {
        char *s = r_str(f);
        out_vec.push_back(std::string(s));
        free(s);
    }
}

void read_array_i32(FILE *f, std::vector<int32_t> & out_vec) {
    uint32_t elem_t = r_u32(f);
    uint64_t n = r_u64(f);
    if (elem_t != GGUF_TYPE_INT32)
        LLM_ERROR("expected int32_t element type but got %d\n", elem_t);
    for (uint64_t i = 0; i < n; ++i) {
        int32_t val = r_i32(f);
        out_vec.push_back(val);
    }
}

void read_array_f32(FILE *f, std::vector<float> & out_vec) {
    uint32_t elem_t = r_u32(f);
    uint64_t n = r_u64(f);
    if (elem_t != GGUF_TYPE_FLOAT32)
        LLM_ERROR("expected float32 element type but got %u\n", elem_t);
    for (uint64_t i = 0; i < n; ++i) {
        float val = r_f32(f);
        out_vec.push_back(val);
    }
}

// print one scalar value (and fully consume bytes)
void print_scalar(FILE *f, uint32_t t, bool is_print) {
    if (!is_print) {
        switch (t){
            case GGUF_TYPE_UINT8:   r_u8(f);  break;
            case GGUF_TYPE_INT8:    r_i8(f);  break;
            case GGUF_TYPE_UINT16:  r_u16(f); break;
            case GGUF_TYPE_INT16:   r_i16(f); break;
            case GGUF_TYPE_UINT32:  r_u32(f); break;
            case GGUF_TYPE_INT32:   r_i32(f); break;
            case GGUF_TYPE_FLOAT32: r_f32(f); break;
            case GGUF_TYPE_BOOL:    r_u8(f);  break;
            case GGUF_TYPE_STRING:  {
                char *s = r_str(f);
                free(s);
            } break;
            case GGUF_TYPE_UINT64:  r_u64(f); break;
            case GGUF_TYPE_INT64:   r_i64(f); break;
            case GGUF_TYPE_FLOAT64: r_f64(f); break;
            default:
                LLM_ERROR("unknown scalar type %u\n", t);
        }
        return;
    }
    switch (t){
        case GGUF_TYPE_UINT8:   printf("%u\n",  (unsigned)r_u8(f)); break;
        case GGUF_TYPE_INT8:    printf("%d\n",  (int)r_i8(f)); break;
        case GGUF_TYPE_UINT16:  printf("%u\n",  (unsigned)r_u16(f)); break;
        case GGUF_TYPE_INT16:   printf("%d\n",  (int)r_i16(f)); break;
        case GGUF_TYPE_UINT32:  printf("%u\n",  r_u32(f)); break;
        case GGUF_TYPE_INT32:   printf("%d\n",  r_i32(f)); break;
        case GGUF_TYPE_FLOAT32: printf("%g\n",  r_f32(f)); break;
        case GGUF_TYPE_BOOL:    printf("%s\n",  r_u8(f) ? "true":"false"); break;
        case GGUF_TYPE_STRING: {
            char *s = r_str(f);
            printf("\"%s\"\n", s);
            free(s);
        } break;
        case GGUF_TYPE_UINT64:  printf("%" PRIu64 "\n", r_u64(f)); break;
        case GGUF_TYPE_INT64:   printf("%" PRId64 "\n", r_i64(f)); break;
        case GGUF_TYPE_FLOAT64: printf("%.17g\n", r_f64(f)); break;
        default:
            LLM_ERROR("unknown scalar type %u\n", t);
    }
}


// print (and consume) an array value
void print_array(FILE *f, bool is_print) {
    uint32_t elem_t = r_u32(f);
    uint64_t n = r_u64(f);
    if (is_print) printf("[");
    for (uint64_t i = 0; i < n; ++i){
        if (i && is_print) printf(", ");
        if (i >= 10) {
            if (is_print) printf("... (total=%" PRIu64 ")]\n", n);
            
            for (; i < n; ++i) {
                if (elem_t == GGUF_TYPE_STRING) {
                    char *tmp = r_str(f);
                    free(tmp);
                } else {
                    switch (elem_t){
                        case GGUF_TYPE_UINT8:  r_u8(f);  break;
                        case GGUF_TYPE_INT8:   r_i8(f);  break;
                        case GGUF_TYPE_UINT16: r_u16(f); break;
                        case GGUF_TYPE_INT16:  r_i16(f); break;
                        case GGUF_TYPE_UINT32: r_u32(f); break;
                        case GGUF_TYPE_INT32:  r_i32(f); break;
                        case GGUF_TYPE_FLOAT32:r_f32(f); break;
                        case GGUF_TYPE_BOOL:   r_u8(f);  break;
                        case GGUF_TYPE_UINT64: r_u64(f); break;
                        case GGUF_TYPE_INT64:  r_i64(f); break;
                        case GGUF_TYPE_FLOAT64:r_f64(f); break;
                        default: LLM_ERROR("unknown array elem type %u\n", elem_t);
                    }
                }
            }
            return;
        }
        if (elem_t == GGUF_TYPE_ARRAY || elem_t == GGUF_TYPE_COUNT)
            LLM_ERROR("invalid nested array element type %u\n", elem_t);
        if (elem_t == GGUF_TYPE_STRING){
            char *s = r_str(f);
            if (is_print) printf("\"%s\"", s);
            free(s);
        } else {
            if (!is_print) {
                switch (elem_t){
                    case GGUF_TYPE_UINT8:  r_u8(f);  break;
                    case GGUF_TYPE_INT8:   r_i8(f);  break;
                    case GGUF_TYPE_UINT16: r_u16(f); break;
                    case GGUF_TYPE_INT16:  r_i16(f); break;
                    case GGUF_TYPE_UINT32: r_u32(f); break;
                    case GGUF_TYPE_INT32:  r_i32(f); break;
                    case GGUF_TYPE_FLOAT32:r_f32(f); break;
                    case GGUF_TYPE_BOOL:   r_u8(f);  break;
                    case GGUF_TYPE_UINT64: r_u64(f); break;
                    case GGUF_TYPE_INT64:  r_i64(f); break;
                    case GGUF_TYPE_FLOAT64:r_f64(f); break;
                    default: LLM_ERROR("unknown array elem type %u\n", elem_t);
                }
                continue;
            }
            // reuse scalar printer by “pretending” file is positioned correctly
            switch (elem_t){
                case GGUF_TYPE_UINT8:   printf("%u",  (unsigned)r_u8(f)); break;
                case GGUF_TYPE_INT8:    printf("%d",  (int)r_i8(f)); break;
                case GGUF_TYPE_UINT16:  printf("%u",  (unsigned)r_u16(f)); break;
                case GGUF_TYPE_INT16:   printf("%d",  (int)r_i16(f)); break;
                case GGUF_TYPE_UINT32:  printf("%u",  r_u32(f)); break;
                case GGUF_TYPE_INT32:   printf("%d",  r_i32(f)); break;
                case GGUF_TYPE_FLOAT32: printf("%g",  r_f32(f)); break;
                case GGUF_TYPE_BOOL:    printf("%s",  r_u8(f) ? "true":"false"); break;
                case GGUF_TYPE_UINT64:  printf("%" PRIu64, r_u64(f)); break;
                case GGUF_TYPE_INT64:   printf("%" PRId64, r_i64(f)); break;
                case GGUF_TYPE_FLOAT64: printf("%.17g", r_f64(f)); break;
                case GGUF_TYPE_STRING:  /* handled above */ break;
                default: LLM_ERROR("unknown array elem type %u\n", elem_t);
            }
        }
    }
    if(is_print) printf("]\n");
}
