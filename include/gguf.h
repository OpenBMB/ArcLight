/**
 * @file gguf.h
 * @brief GGUF file format reader
 * 
 * Provides:
 *  - Basic utilities for reading GGUF files, including scalar and array types.
 *  - Functions to read arrays of specific types into C++ vectors.
 * 
 * Designed for read model metadata and parameters stored in GGUF format, with a focus on simplicity and correctness.
 */
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>


#define GGUF_GENERAL_ALIGNMENT 32
#define GGUF_ALIGN_UP(x)  (((x) + (GGUF_GENERAL_ALIGNMENT - 1)) & ~(GGUF_GENERAL_ALIGNMENT - 1))

// #define LLM_ERROR(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(1); } while(0)
// #define LLM_LOG(is_print, ...) do { if (is_print) { fprintf(stdout, __VA_ARGS__); fputc('\n', stdout); } } while(0)
#define TRYREAD(ptr, sz, n, f) do { if (fread((ptr), (sz), (n), (f)) != (n)) LLM_ERROR("short read"); } while(0)

#define LLM_LOG_PRINT(...) do { fprintf(stdout, __VA_ARGS__); } while(0)
#define LLM_LOG(is_print, ...) do { if (is_print) { fprintf(stdout, __VA_ARGS__); } } while(0)
#define LLM_ERROR(...) do { fprintf(stderr, __VA_ARGS__); exit(1); } while(0)

// GGUF value types (from spec)
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT   = 13
};


static uint8_t  r_u8 (FILE *f){ uint8_t  x; TRYREAD(&x,1,1,f); return x; }
static int8_t   r_i8 (FILE *f){ int8_t   x; TRYREAD(&x,1,1,f); return x; }
static uint16_t r_u16(FILE *f){ uint16_t x; TRYREAD(&x,2,1,f); return x; }
static int16_t  r_i16(FILE *f){ int16_t  x; TRYREAD(&x,2,1,f); return x; }
static uint32_t r_u32(FILE *f){ uint32_t x; TRYREAD(&x,4,1,f); return x; }
static int32_t  r_i32(FILE *f){ int32_t  x; TRYREAD(&x,4,1,f); return x; }
static uint64_t r_u64(FILE *f){ uint64_t x; TRYREAD(&x,8,1,f); return x; }
static int64_t  r_i64(FILE *f){ int64_t  x; TRYREAD(&x,8,1,f); return x; }
static float    r_f32(FILE *f){ float    x; TRYREAD(&x,4,1,f); return x; }
static double   r_f64(FILE *f){ double   x; TRYREAD(&x,8,1,f); return x; }


static char *r_str(FILE *f){
    uint64_t n = r_u64(f);
    if (n > (1ULL<<31)) LLM_ERROR("string too large: %" PRIu64 "\n", n);
    char *s = (char*)malloc((size_t)n + 1);
    if (!s) LLM_ERROR("malloc fail\n");
    TRYREAD(s, 1, (size_t)n, f);
    s[n] = '\0'; // gguf strings are NOT null-terminated on disk
    return s;
}

void read_array_str(FILE *f, std::vector<std::string> & out_vec);
void read_array_i32(FILE *f, std::vector<int32_t> & out_vec);
void read_array_f32(FILE *f, std::vector<float> & out_vec);

void print_scalar(FILE *f, uint32_t t, bool is_print);
void print_array(FILE *f, bool is_print);
