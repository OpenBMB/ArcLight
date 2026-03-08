/**
 * @file tensor.h
 * @brief tensor abstraction and basic operations for NNML.
 * 
 * Provides:
 *  - A tensor class with up to 4 dimensions, supporting various data types and tensor types.
 *  - Basic tensor operations such as reshape and view, with support for non-contiguous tensors.
 *  - Utility functions for checking tensor properties (e.g., contiguity, shape compatibility).
 * 
 * Designed for inference workloads with a focus on efficient memory layout and simplified management.
 */
#pragma once

#include <string>
#include <assert.h>
#include <vector>
#include <cstdarg>

#include "nnml.h"
#include "ops.h"


#define NNML_MAX_DIMS           4
#define NNML_MAX_OP_PARAMS      64
#define NNML_MAX_SRC            16
#define NNML_MAX_NAME           64

#define NNML_ROPE_TYPE_NEOX   2
#define NNML_ROPE_TYPE_MROPE  8

#define NNML_MROPE_SECTIONS   4

#define NNML_KQ_MASK_PAD 64

#if UINTPTR_MAX == 0xFFFFFFFF
    #define NNML_MEM_ALIGN 4
#else
    #define NNML_MEM_ALIGN 16
#endif
#define NNML_ALIGN_UP(x)  (((x) + (NNML_MEM_ALIGN - 1)) & ~(NNML_MEM_ALIGN - 1))


enum nnml_tensor_type {
        NNML_TENSOR_TYPE_WEIGHT     =  1,
        NNML_TENSOR_TYPE_ACTIVATION =  2,
        NNML_TENSOR_TYPE_DUAL_ACTI  =  3,
        NNML_TENSOR_TYPE_KVCACHE    =  4,
    };

struct nnml_shape {
    std::array<int64_t, NNML_MAX_DIMS> ne{ {1,1,1,1} };
    int                                n_dims = 1;

    int64_t dim(int i) const noexcept {
        NNML_ASSERT_MSG(i >= 0 && i < n_dims, "dim index out of bounds of shape of tensor");
        return ne[static_cast<size_t>(i)];
    }
    int64_t elements() const noexcept {
        int64_t e = 1;
        for (int i = 0; i < n_dims; ++i) e *= ne[static_cast<size_t>(i)];
        return e;
    }
};

struct nnml_stride {
    std::array<size_t, NNML_MAX_DIMS> nb{ {0,0,0,0} };
};


/**
 * nnml_tensor: a simple tensor class that is encapsulated.
 *  - Supports up to 4 dimensions, with flexible shape and stride.
 *  - Contains metadata such as data type, tensor type, operation, and source tensors.
 *  - Designed for efficient memory layout and easy integration with nnml_memory.
 *  - Does not manage its own memory; relies on nnml_memory for allocation.
 *  - Provides utility functions for checking properties like contiguity and shape compatibility.
 */
class nnml_tensor {
public:
    nnml_tensor() noexcept = default;
    ~nnml_tensor();

    static nnml_tensor * new_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int n_dims, const int64_t *ne, nnml_tensor * view_src, size_t view_offs);
    static nnml_tensor * new_tensor(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int n_dims, const int64_t *ne);
    static nnml_tensor * new_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0);
    static nnml_tensor * new_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1);
    static nnml_tensor * new_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
    static nnml_tensor * new_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

    static nnml_tensor * view(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * src);
    static nnml_tensor * duplicate(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const nnml_tensor * src);

    static bool are_same_shape(const nnml_tensor * t0, const nnml_tensor * t1);
    static bool are_same_stride(const nnml_tensor * t0, const nnml_tensor * t1);
    static bool can_repeat(nnml_tensor * t0, nnml_tensor * t1);
    static bool can_repeat_rows(nnml_tensor * t0, nnml_tensor * t1);

    nnml_type           get_data_type()                 const noexcept { return type; }
    void                set_data_type(nnml_type t)            noexcept { type = t; }
    nnml_tensor_type    get_tensor_type()               const noexcept { return tensor_type; }
    void                set_tensor_type(nnml_tensor_type t)   noexcept { tensor_type = t; }
    nnml_op             get_operation()                 const noexcept { return op; }
    void                set_operation(nnml_op o)              noexcept { op = o; }
    int                 n_dims()                        const noexcept;
    nnml_shape          shape()                         const noexcept { nnml_shape s; s.ne = load_ne_(); s.n_dims = n_dims(); return s; }
    nnml_stride         stride()                        const noexcept { nnml_stride s; s.nb = load_nb_(); return s; }
    int64_t             n_rows()                        const noexcept { return ne[1]*ne[2]*ne[3]; }
    int64_t             n_elements()                    const noexcept { return ne[0]*ne[1]*ne[2]*ne[3]; }
    int64_t             get_elements(int i)             const noexcept { return ne[i]; }
    void                set_elements(int i, int64_t v)        noexcept { ne[i] = v; }
    size_t              get_stride_bytes(int i)         const noexcept { return nb[i]; }
    void                set_stride_bytes(int i, size_t v)     noexcept { nb[i] = v; }
    size_t              element_size()                  const noexcept { return nnml_type_size(this->type); }
    void *              tensor_data()                   const noexcept { return data; }
    void                set_tensor_data(void * ptr)           noexcept { data = ptr; }
    void                copy_from(void * from, size_t offset, size_t nbytes) noexcept;
    int64_t *           get_ne_ptr()                          noexcept { return ne; }
    size_t *            get_nb_ptr()                          noexcept { return nb; }
    char *              get_name_ptr()                        noexcept { return name; }
    const char *        get_name_cstr()                 const noexcept { return name; }
    int32_t             get_tensor_flags()              const noexcept { return flags; }
    nnml_tensor *       get_src_tensor(int i)           const noexcept { return src[i]; }
    void                set_src_tensor(int i, nnml_tensor * t)noexcept { src[i] = t; }
    const int32_t *     get_operation_params()          const noexcept { return op_params; }
    size_t              get_view_offset()               const noexcept { return view_offs; }
    nnml_tensor *       get_view_src()                  const noexcept { return view_src; }
    bool                is_view()                       const noexcept { return view_src != nullptr; }
    bool                is_scalar()                     const noexcept;
    bool                is_vector()                     const noexcept;
    bool                is_matrix()                     const noexcept;
    bool                is_3d()                         const noexcept;
    bool                is_transposed()                 const noexcept;
    bool                is_contiguous_n(int n)          const noexcept;
    bool                is_contiguous()                 const noexcept;
    bool                is_contiguous_0()               const noexcept;
    bool                is_contiguous_1()               const noexcept;
    bool                is_contiguous_2()               const noexcept;
    bool                is_contiguously_allocated()     const noexcept;
    bool                is_permuted()                   const noexcept;
    bool                is_contiguous_channels()        const noexcept;
    bool                is_contiguous_rows()            const noexcept;
    bool                is_empty()                      const noexcept;
    bool                is_padded_1d();
    nnml_tensor&        set_name(const char * fmt, ...);
    nnml_tensor&        set_flags(int32_t f);              // no implementation now
    void                set_op_params(const void * params, size_t params_size);
    void                set_op_params_i32(uint32_t i, int32_t value);
    void                set_op_params_f32(uint32_t i, float value);
    int32_t             get_op_params_i32(uint32_t i)   const;
    float               get_op_params_f32(uint32_t i)   const;
    nnml_glu_op         get_glu_op()                    const;
    void                set_asm_gemm()                  noexcept       { extra[0] = 1; }
    bool                is_asm_gemm()                   const noexcept { return extra[0] != 0; }
    void                print_data(uint32_t max_elements = 12, bool all = true, int32_t start_idx = 0) const;
    void                save_data(const char * filename) const;
    void                load_data(const char * filename);
    bool                compare_from(const char * filename) const;    

private:
    // note: the following is innitialized by nnml_tensor::new_impl
    enum nnml_type type;
    enum nnml_tensor_type tensor_type;

    int64_t ne[NNML_MAX_DIMS]; // number of elements
    size_t  nb[NNML_MAX_DIMS]; // stride in bytes:
                               // nb[0] = nnml_type_size(type)
                               // nb[1] = nb[0]   * (ne[0] / nnml_blck_size(type)) + padding
                               // nb[i] = nb[i-1] * ne[i-1]

    enum nnml_op op;
    int32_t op_params[NNML_MAX_OP_PARAMS / sizeof(int32_t)];    // op params
    int32_t flags;

    struct nnml_tensor * src[NNML_MAX_SRC];
    // source tensor and offset for views
    struct nnml_tensor * view_src;
    size_t               view_offs;

    void * data;

    char name[NNML_MAX_NAME];
    int  extra[4]; // extra[0]: is asm gemm, extra[1:3]: not defined
    char padding[8];


    // inline in-class helpers
    // convert ne/nb to std::array
    std::array<int64_t, NNML_MAX_DIMS> load_ne_() const noexcept {
        std::array<int64_t, NNML_MAX_DIMS> out{};
        for (int i = 0; i < NNML_MAX_DIMS; ++i) out[static_cast<size_t>(i)] = ne[i];
        return out;
    }
    std::array<size_t, NNML_MAX_DIMS> load_nb_() const noexcept {
        std::array<size_t, NNML_MAX_DIMS> out{};
        for (int i = 0; i < NNML_MAX_DIMS; ++i) out[static_cast<size_t>(i)] = nb[i];
        return out;
    }
};

static const size_t NNML_TENSOR_SIZE = NNML_ALIGN_UP(sizeof(class nnml_tensor));


inline nnml_tensor * tensor_new_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int n_dims, const int64_t *ne, nnml_tensor * view_src, size_t view_offs) {
    return nnml_tensor::new_impl(mem, tensor_type, buffer_id, dual_idx, type, n_dims, ne, view_src, view_offs);
}
inline nnml_tensor * tensor_new(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int n_dims, const int64_t *ne) {
    return nnml_tensor::new_tensor(mem, tensor_type, buffer_id, dual_idx, type, n_dims, ne);
}
inline nnml_tensor * tensor_new_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int64_t n0) {
    return nnml_tensor::new_1d(mem, tensor_type, buffer_id, dual_idx, type, n0);
}
inline nnml_tensor * tensor_new_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int64_t n0, int64_t n1) {
    return nnml_tensor::new_2d(mem, tensor_type, buffer_id, dual_idx, type, n0, n1);
}
inline nnml_tensor * tensor_new_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int64_t n0, int64_t n1, int64_t n2) {
    return nnml_tensor::new_3d(mem, tensor_type, buffer_id, dual_idx, type, n0, n1, n2);
}
inline nnml_tensor * tensor_new_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_type type, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return nnml_tensor::new_4d(mem, tensor_type, buffer_id, dual_idx, type, n0, n1, n2, n3);
}

inline nnml_tensor * tensor_view (nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_tensor * a) {
    return nnml_tensor::view(mem, tensor_type, buffer_id, dual_idx, a);
}
inline nnml_tensor * tensor_dup (nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        const nnml_tensor * a) {
    return nnml_tensor::duplicate(mem, tensor_type, buffer_id, dual_idx, a);
}

inline bool are_same_shape(const nnml_tensor * t0, const nnml_tensor * t1) {
    return nnml_tensor::are_same_shape(t0, t1);
}
inline bool are_same_stride(const nnml_tensor * t0, const nnml_tensor * t1) {
    return nnml_tensor::are_same_stride(t0, t1);
}
inline bool can_repeat(nnml_tensor * t0, nnml_tensor * t1) {
    return nnml_tensor::can_repeat(t0, t1);
}
inline bool can_repeat_rows(nnml_tensor * t0, nnml_tensor * t1) {
    return nnml_tensor::can_repeat_rows(t0, t1);
}


/**
 * Utility functions for tensor operations and properties.
 */
size_t nnml_row_size(enum nnml_type type, int64_t ne);
size_t nnml_nbytes(const nnml_tensor * tensor);
nnml_tensor * nnml_pad_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3);
nnml_tensor * nnml_pad(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int p0, int p1, int p2, int p3);
nnml_tensor * nnml_get_rows(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_set_rows(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c);

static nnml_tensor * nnml_unary_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, enum nnml_unary_op op, bool inplace);
nnml_tensor * nnml_unary(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, enum nnml_unary_op op);
nnml_tensor * nnml_unary_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, enum nnml_unary_op op);

static nnml_tensor * nnml_dup_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, bool inplace);
nnml_tensor * nnml_dup(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_dup_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

static nnml_tensor * nnml_add_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, bool inplace);
nnml_tensor * nnml_add(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_add_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);

static nnml_tensor * nnml_sub_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, bool inplace);
nnml_tensor * nnml_sub(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_sub_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);

static nnml_tensor * nnml_mul_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, bool inplace);
nnml_tensor * nnml_mul(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_mul_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);

static nnml_tensor * nnml_div_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, bool inplace);
nnml_tensor * nnml_div(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_div_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);

static nnml_tensor * nnml_sqr_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, bool inplace);
nnml_tensor * nnml_sqr(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_sqr_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

static nnml_tensor * nnml_sqrt_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, bool inplace);
nnml_tensor * nnml_sqrt(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_sqrt_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

static nnml_tensor * nnml_log_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, bool inplace);
nnml_tensor * nnml_log(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_log_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

bool nnml_can_mul_mat(const struct nnml_tensor * t0, const struct nnml_tensor * t1);
nnml_tensor * nnml_mul_mat(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
void nnml_mul_mat_set_prec(nnml_tensor * a, nnml_prec prec);

static nnml_tensor * nnml_scale_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float s, float b, bool inplace);
nnml_tensor * nnml_scale(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float s);
nnml_tensor * nnml_scale_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float s);
nnml_tensor * nnml_scale_bias(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float s, float b);
nnml_tensor * nnml_scale_bias_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float s, float b);

nnml_tensor * nnml_cast(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_type type);

static nnml_tensor * nnml_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps, bool inplace);
nnml_tensor * nnml_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps);
nnml_tensor * nnml_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps);

static nnml_tensor * nnml_rms_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps, bool inplace);
nnml_tensor * nnml_rms_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps);
nnml_tensor * nnml_rms_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps);
nnml_tensor * nnml_rms_norm_back(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, float eps);

static nnml_tensor * nnml_group_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int n_groups, float eps, bool inplace);
nnml_tensor * nnml_group_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int n_groups, float eps);
nnml_tensor * nnml_group_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int n_groups, float eps);

nnml_tensor * nnml_reshape(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_reshape_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0);
nnml_tensor * nnml_reshape_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1);
nnml_tensor * nnml_reshape_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
nnml_tensor * nnml_reshape_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

static nnml_tensor * nnml_view_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int n_dims, const int64_t *ne, size_t offset);
nnml_tensor * nnml_view_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, size_t offset);
nnml_tensor * nnml_view_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
nnml_tensor * nnml_view_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
nnml_tensor * nnml_view_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

nnml_tensor * nnml_permute(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int axis0, int axis1, int axis2, int axis3);
nnml_tensor * nnml_transpose(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

static nnml_tensor * nnml_rope_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, nnml_tensor * c, int n_dims, int sections[NNML_MROPE_SECTIONS], int mode,
    int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, bool inplace);
nnml_tensor * nnml_rope(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, int n_dims, int mode);
nnml_tensor * nnml_rope_multi(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int sections[NNML_MROPE_SECTIONS], int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow);
nnml_tensor * nnml_rope_multi_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int sections[NNML_MROPE_SECTIONS], int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
nnml_tensor * nnml_rope_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, int n_dims, int mode);
nnml_tensor * nnml_rope_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
nnml_tensor * nnml_rope_ext_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

static nnml_tensor * nnml_glu_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, nnml_glu_op op, bool swapped);
nnml_tensor * nnml_swiglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_swiglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_geglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_geglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_reglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_reglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b);
nnml_tensor * nnml_relu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_relu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_silu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_silu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_gelu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_gelu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_tanh(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_tanh_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);

static nnml_tensor * nnml_soft_max_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    nnml_tensor * mask, float scale, float max_bias, bool inplace);
nnml_tensor * nnml_soft_max(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_soft_max_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_soft_max_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * mask, float scale, float max_bias);
void nnml_soft_max_add_sinks(nnml_tensor * a, nnml_tensor * sinks);

static nnml_tensor * nnml_cont_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_cont(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_cont_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
nnml_tensor * nnml_cont_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0);
nnml_tensor * nnml_cont_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1);
nnml_tensor * nnml_cont_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);

nnml_tensor * nnml_scatter_prepare(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_scatter_copy(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a);
nnml_tensor * nnml_gather_copy(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor ** a, int n_tensors);

nnml_tensor * nnml_flash_attn_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * q, nnml_tensor * k, nnml_tensor * v,
    nnml_tensor * mask, float scale, float max_bias, float logit_softcap);
void nnml_flash_attn_ext_add_sinks(nnml_tensor * a, nnml_tensor * sinks);
void nnml_flash_attn_ext_set_prec(nnml_tensor * a, nnml_prec prec);
