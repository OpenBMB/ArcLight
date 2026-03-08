/**
 * @file tensor.cpp
 * @brief Implementation of tensor.h
 * 
 * This file implements the nnml_tensor class and related functions for tensor operations and properties.
 */
#include <cstdarg>

#include "tensor.h"
#include "memory.h"

// implementation of nnml_tensor
nnml_tensor& nnml_tensor::set_name(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(name, sizeof(name), fmt, args);
    va_end(args);
    return *this;
}

void nnml_tensor::set_op_params(const void * params, size_t params_size) {
    assert(params_size <= NNML_MAX_OP_PARAMS);
    memcpy(op_params, params, params_size);
}

void nnml_tensor::set_op_params_i32(uint32_t i, int32_t value) {
    assert(i < NNML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(op_params))[i] = value;
}

void nnml_tensor::set_op_params_f32(uint32_t i, float value) {
    assert(i < NNML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(op_params))[i] = value;
}

int32_t nnml_tensor::get_op_params_i32(uint32_t i) const {
    assert(i < NNML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((int32_t *)(op_params))[i];
}

float nnml_tensor::get_op_params_f32(uint32_t i) const {
    assert(i < NNML_MAX_OP_PARAMS / sizeof(float));
    return ((float *)(op_params))[i];
}

nnml_glu_op nnml_tensor::get_glu_op() const {
    NNML_ASSERT(this->get_operation() == NNML_OP_GLU);
    return (nnml_glu_op) this->get_op_params_i32(0);
}

void nnml_tensor::print_data(uint32_t max_elements, bool all, int32_t start_idx) const {
    if (all) {
        const size_t nbytes = nnml_nbytes(this);
        const size_t type_size = nnml_type_size(this->get_data_type());
        const int32_t n_elements = nbytes / type_size;
        if (start_idx >= n_elements) {
            printf("Tensor %s: start_idx %d out of bounds (size %d)\n", name, start_idx, n_elements);
            return;
        }
        const int32_t n_remaining = n_elements - start_idx;
        const int32_t n_print = std::min((int32_t)max_elements, n_remaining);

        printf("Tensor %s: type=%d, n_dims=%d, ne=[", name, (int)this->get_data_type(), this->n_dims());
        for (int i = 0; i < this->n_dims(); ++i) {
            printf("%s%lld", i == 0 ? "" : ", ", (long long)this->get_elements(i));
        }
        printf("], data (offset %d, count %d): ", start_idx, n_print);

        for (size_t i = 0; i < n_print; ++i) {
            int32_t current_idx = start_idx + i;
            if (i > 0) {
                printf(", ");
            }
            switch (this->get_data_type()) {
                case NNML_TYPE_F32: {
                    float * data_f32 = (float *) this->tensor_data();
                    printf("%.4f", data_f32[current_idx]);
                } break;
                case NNML_TYPE_F16: {
                    nnml_fp16_t * data_f16 = (nnml_fp16_t *) this->tensor_data();
                    float v;
                    nnml_cpu_fp16_to_fp32(&data_f16[current_idx], &v, 1);
                    printf("%.4f", v);
                } break;
                case NNML_TYPE_I32: {
                    int32_t * data_i32 = (int32_t *) this->tensor_data();
                    printf("%d", data_i32[current_idx]);
                } break;
                case NNML_TYPE_I64: {
                    int64_t * data_i64 = (int64_t *) this->tensor_data();
                    printf("%lld", (long long)data_i64[current_idx]);
                } break;
                default:
                    printf("?");
                    break;
            }
        }
        printf("\n");
    } else {
        printf("data (first %u bytes): ", max_elements);
        uint8_t * data = (uint8_t *) this;
        const int32_t n_remaining = nnml_nbytes(this) - start_idx;
        const int32_t n_print = std::min((int32_t)max_elements, n_remaining);
        for (int32_t i = start_idx; i < n_print+start_idx; ++i) {
            // print the max_elements bytes in hex
            printf("%02x ", data[i]);
        }
        printf("\n");
    }
}

void nnml_tensor::save_data(const char * filename) const {
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }
    size_t nbytes = nnml_nbytes(this);
    size_t written = fwrite(this->tensor_data(), 1, nbytes, f);
    if (written != nbytes) {
        fprintf(stderr, "Failed to write all data to file %s\n", filename);
    }
    fclose(f);
}

void nnml_tensor::load_data(const char * filename) {
    FILE * f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file %s for reading\n", filename);
        return;
    }
    size_t nbytes = nnml_nbytes(this);
    size_t read = fread(this->tensor_data(), 1, nbytes, f);
    if (read != nbytes) {
        fprintf(stderr, "Failed to read all data from file %s\n", filename);
    }
    fclose(f);
}

bool nnml_tensor::compare_from(const char * filename) const {
    FILE * f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file %s for reading\n", filename);
        return false;
    }
    size_t nbytes = nnml_nbytes(this);
    uint8_t * buffer = new uint8_t[nbytes];
    size_t read = fread(buffer, 1, nbytes, f);
    if (read != nbytes) {
        fprintf(stderr, "Failed to read all data from file %s\n", filename);
        delete[] buffer;
        fclose(f);
        return false;
    }
    fclose(f);

    bool match = memcmp(this->tensor_data(), buffer, nbytes) == 0;
    if (!match) {
        fprintf(stderr, "Data does not match for file %s\n", filename);
    }
    delete[] buffer;
    return match;
}

void nnml_tensor::copy_from(void * from, size_t offset, size_t nbytes) noexcept {
    assert(from != nullptr);
    assert(data != nullptr);
    assert(offset + nbytes <= nnml_nbytes(this));
    memcpy(static_cast<uint8_t *>(data) + offset, from, nbytes);
}

nnml_tensor * nnml_tensor::new_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_type type, int n_dims, const int64_t *ne, nnml_tensor * view_src, size_t view_offs) {
    NNML_ASSERT(type >= 0 && type < NNML_TYPE_COUNT);
    NNML_ASSERT(n_dims >= 1 && n_dims <= NNML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = nnml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    // printf("view src: %p, view_offs: %zu, data_size: %zu\n", (void *)view_src, view_offs, data_size);
    NNML_ASSERT(view_src == NULL || data_size == 0 || view_offs + data_size <= nnml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    // here we just do not allow no_alloc property
    if (view_src == NULL) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    void * const obj_new = mem->allocate_obj(tensor_type, buffer_id, dual_idx, NNML_TENSOR_SIZE + obj_alloc_size);
    NNML_ASSERT(obj_new);
    nnml_tensor * const result = (nnml_tensor *)obj_new;

    result->type         = type;
    result->tensor_type  = tensor_type;
    for (int i = 0; i < NNML_MAX_DIMS; ++i) result->ne[i] = 1;
    for (int i = 0; i < NNML_MAX_DIMS; ++i) result->nb[i] = 0;
    result->op           = NNML_OP_NONE;
    memset(result->op_params, 0, sizeof(result->op_params));
    result->flags        = 0;
    for (int i = 0; i < NNML_MAX_SRC; ++i) result->src[i] = nullptr;
    result->view_src     = view_src;
    result->view_offs    = view_offs;
    result->data         = obj_alloc_size > 0 ? (void *)((uint8_t *)obj_new + NNML_TENSOR_SIZE) : data;
    memset(result->name, 0, sizeof(result->name));
    // result->extra        = nullptr;
    for (int i = 0; i < 4; ++i) result->extra[i] = 0;
    memset(result->padding, 0, sizeof(result->padding));

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = nnml_type_size(type);
    result->nb[1] = result->nb[0] * (result->ne[0] / nnml_blck_size(type));
    for (int i = 2; i < NNML_MAX_DIMS; ++i) {
        result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
    }
    
    // ctx->n_objects++;
    return result;
}

nnml_tensor * nnml_tensor::new_tensor(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int n_dims, const int64_t *ne) {
    return new_impl(mem, tensor_type, buffer_id, dual_idx, type, n_dims, ne, nullptr, 0);
}

nnml_tensor * nnml_tensor::new_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0) {
    return new_impl(mem, tensor_type, buffer_id, dual_idx, type, 1, &ne0, nullptr, 0);
}

nnml_tensor * nnml_tensor::new_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return new_impl(mem, tensor_type, buffer_id, dual_idx, type, 2, ne, nullptr, 0);
}

nnml_tensor * nnml_tensor::new_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return new_impl(mem, tensor_type, buffer_id, dual_idx, type, 3, ne, nullptr, 0);
}

nnml_tensor * nnml_tensor::new_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return new_impl(mem, tensor_type, buffer_id, dual_idx, type, 4, ne, nullptr, 0);
}

nnml_tensor * nnml_tensor::duplicate(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, const nnml_tensor * src) {
    return new_impl(mem, tensor_type, buffer_id, dual_idx, src->type, NNML_MAX_DIMS, src->ne, nullptr, 0);
}

nnml_tensor * nnml_tensor::view(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * src) {
    nnml_tensor * result = new_impl(mem, tensor_type, buffer_id, dual_idx, src->type, NNML_MAX_DIMS, src->ne, src, 0);
    result->set_name("%s (view)", src->name);
    for (int i = 0; i < NNML_MAX_DIMS; ++i) result->nb[i] = src->nb[i];
    return result;
}

bool nnml_tensor::is_scalar() const noexcept {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return ne[0] == 1 && ne[1] == 1 && ne[2] == 1 && ne[3] == 1;
}

bool nnml_tensor::is_vector() const noexcept {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return ne[1] == 1 && ne[2] == 1 && ne[3] == 1;
}

bool nnml_tensor::is_matrix() const noexcept {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return ne[2] == 1 && ne[3] == 1;
}

bool nnml_tensor::is_3d() const noexcept {
    return ne[3] == 1;
}

int nnml_tensor::n_dims() const noexcept {
    for (int i = NNML_MAX_DIMS - 1; i >= 1; --i) {
        if (ne[i] > 1) {
            return i + 1;
        }
    }
    return 1;
}

bool nnml_tensor::is_transposed() const noexcept {
    return nb[0] > nb[1];
}

bool nnml_tensor::is_contiguous_n(int n) const noexcept {
    size_t next_nb = nnml_type_size(type);
    if (ne[0] != nnml_blck_size(type) && nb[0] != next_nb) {
        return false;
    }
    next_nb *= ne[0] / nnml_blck_size(type);
    for (int i = 1; i < NNML_MAX_DIMS; i++) {
        if (ne[i] != 1) {
            if (i > n) {
                if (nb[i] != next_nb) {
                    return false;
                }
                next_nb *= ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = ne[i] * nb[i];
            }
        }
    }
    return true;
}

bool nnml_tensor::is_contiguous() const noexcept {
    return is_contiguous_0();
}

bool nnml_tensor::is_contiguous_0() const noexcept {
    return is_contiguous_n(0);
}

bool nnml_tensor::is_contiguous_1() const noexcept {
    return is_contiguous_n(1);
}

bool nnml_tensor::is_contiguous_2() const noexcept {
    return is_contiguous_n(2);
}

bool nnml_tensor::is_contiguously_allocated() const noexcept {
    return nnml_nbytes(this) == n_elements() * nnml_type_size(type) / nnml_blck_size(type);
}

bool nnml_tensor::is_permuted() const noexcept {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return nb[0] > nb[1] || nb[1] > nb[2] || nb[2] > nb[3];
}

bool nnml_tensor::is_contiguous_channels() const noexcept {
    return
        nb[0] > nb[2] &&
        nb[1] > nb[0] &&
        nb[2] == nnml_type_size(type);
}

bool nnml_tensor::is_contiguous_rows() const noexcept {
    return
        ne[0] == nnml_blck_size(type) ||
        nb[0] == nnml_type_size(type);
}

bool nnml_tensor::is_empty() const noexcept {
    for (int i = 0; i < NNML_MAX_DIMS; ++i) {
        if (ne[i] == 0) {
            // empty if any dimension has no elements
            return true;
        }
    }
    return false;
}

bool nnml_tensor::are_same_shape(const nnml_tensor * t0, const nnml_tensor * t1) {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool nnml_tensor::are_same_stride(const nnml_tensor * t0, const nnml_tensor * t1) {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool nnml_tensor::can_repeat(nnml_tensor * t0, nnml_tensor * t1) {
    // check if t1 can be represented as a repetition of t0
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return t0->is_empty() ? t1->is_empty() :
        (t1->ne[0] % t0->ne[0] == 0) &&
        (t1->ne[1] % t0->ne[1] == 0) &&
        (t1->ne[2] % t0->ne[2] == 0) &&
        (t1->ne[3] % t0->ne[3] == 0);
}

bool nnml_tensor::can_repeat_rows(nnml_tensor * t0, nnml_tensor * t1) {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return (t0->ne[0] == t1->ne[0]) && can_repeat(t0, t1);
}

bool nnml_tensor::is_padded_1d() {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return
        nb[0] == nnml_type_size(type) &&
        nb[2] == nb[1] * ne[1] &&
        nb[3] == nb[2] * ne[2];
}


// implementation of utility functions for tensor operations and properties
size_t nnml_row_size(enum nnml_type type, int64_t ne) {
    assert(ne % nnml_blck_size(type) == 0);
    return nnml_type_size(type) * ne / nnml_blck_size(type);
}

size_t nnml_nbytes(const nnml_tensor * tensor) {
    for (int i = 0; i < NNML_MAX_DIMS; ++i) {
        if (tensor->get_elements(i) <= 0) {
            return 0;
        }
    }
    size_t nbytes;
    const size_t blck_size = nnml_blck_size(tensor->get_data_type());
    if (blck_size == 1) {
        nbytes = nnml_type_size(tensor->get_data_type());
        for (int i = 0; i < NNML_MAX_DIMS; ++i) {
            nbytes += (tensor->get_elements(i) - 1) * tensor->get_stride_bytes(i);
        }
    }
    else {
        nbytes = tensor->get_elements(0) * tensor->get_stride_bytes(0) / blck_size;
        for (int i = 1; i < NNML_MAX_DIMS; ++i) {
            nbytes += (tensor->get_elements(i) - 1) * tensor->get_stride_bytes(i);
        }
    }
    return nbytes;
}

nnml_tensor * nnml_pad_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3) {
    nnml_tensor * result = tensor_new_4d(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(),
                a->get_elements(0) + lp0 + rp0, a->get_elements(1) + lp1 + rp1, a->get_elements(2) + lp2 + rp2, a->get_elements(3) + lp3 + rp3);
    result->set_op_params_i32(0, lp0);
    result->set_op_params_i32(1, rp0);
    result->set_op_params_i32(2, lp1);
    result->set_op_params_i32(3, rp1);
    result->set_op_params_i32(4, lp2);
    result->set_op_params_i32(5, rp2);
    result->set_op_params_i32(6, lp3);
    result->set_op_params_i32(7, rp3);
    result->set_operation(NNML_OP_PAD);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_pad(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    int p0, int p1, int p2, int p3) {
    return nnml_pad_ext(mem, tensor_type, buffer_id, dual_idx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

nnml_tensor * nnml_get_rows(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    NNML_ASSERT(a->get_elements(2) == b->get_elements(1));
    NNML_ASSERT(a->get_elements(3) == b->get_elements(2));
    NNML_ASSERT(b->get_elements(3) == 1);
    NNML_ASSERT(b->get_data_type() == NNML_TYPE_I32);

    // TODO: implement non F32 return
    enum nnml_type type = NNML_TYPE_F32;
    if (a->get_data_type() == NNML_TYPE_I32) {
        type = a->get_data_type();
    }
    nnml_tensor * result = tensor_new_4d(mem, tensor_type, buffer_id, dual_idx, type,
                            a->get_elements(0), b->get_elements(0), b->get_elements(1), b->get_elements(2));

    result->set_operation(NNML_OP_GET_ROWS);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_set_rows(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c) {
    NNML_ASSERT(a->get_elements(0) == b->get_elements(0));
    NNML_ASSERT(a->get_elements(2) == b->get_elements(2));
    NNML_ASSERT(a->get_elements(3) == b->get_elements(3));
    NNML_ASSERT(b->get_elements(1) == c->get_elements(0));
    NNML_ASSERT(b->get_elements(2) % c->get_elements(1) == 0);
    NNML_ASSERT(b->get_elements(3) % c->get_elements(2) == 0);
    NNML_ASSERT(c->get_elements(3) == 1);
    NNML_ASSERT(b->get_data_type() == NNML_TYPE_F32);
    NNML_ASSERT(c->get_data_type() == NNML_TYPE_I64 || c->get_data_type() == NNML_TYPE_I32);
    NNML_ASSERT(a->is_contiguous_rows());
    NNML_ASSERT(b->is_contiguous_rows());

    nnml_tensor * result = tensor_view(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_SET_ROWS);
    result->set_src_tensor(0, b);
    result->set_src_tensor(1, c);
    result->set_src_tensor(2, a);
    return result;
}

static nnml_tensor * nnml_unary_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_unary_op op, bool inplace) {
    NNML_ASSERT(a->is_contiguous_1());
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_op_params_i32(0, (int32_t) op);
    result->set_operation(NNML_OP_UNARY);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_unary(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_unary_op op) {
    return nnml_unary_impl(mem, tensor_type, buffer_id, dual_idx, a, op, false);
}

nnml_tensor * nnml_unary_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_unary_op op) {
    return nnml_unary_impl(mem, tensor_type, buffer_id, dual_idx, a, op, true);
}

// nnml_dup

static nnml_tensor * nnml_dup_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
        nnml_tensor * a, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a) : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_DUP);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_dup(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_dup_impl(mem, tensor_type, buffer_id, dual_idx, a, false);
}

nnml_tensor * nnml_dup_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_dup_impl(mem, tensor_type, buffer_id, dual_idx, a, true);
}

// nnml_add

static nnml_tensor * nnml_add_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, bool inplace) {
    NNML_ASSERT(can_repeat(b, a));
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_ADD);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_add(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_add_impl(mem, tensor_type, buffer_id, dual_idx, a, b, false);
}

nnml_tensor * nnml_add_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_add_impl(mem, tensor_type, buffer_id, dual_idx, a, b, true);
}

// nnml_sub

static nnml_tensor * nnml_sub_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, bool inplace) {
    NNML_ASSERT(can_repeat(b, a));
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_SUB);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_sub(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_sub_impl(mem, tensor_type, buffer_id, dual_idx, a, b, false);
}

nnml_tensor * nnml_sub_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_sub_impl(mem, tensor_type, buffer_id, dual_idx, a, b, true);
}

// nnml_mul

static nnml_tensor * nnml_mul_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, bool inplace) {
    NNML_ASSERT(can_repeat(b, a));
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_MUL);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_mul(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b) {
    return nnml_mul_impl(mem, tensor_type, buffer_id, dual_idx, a, b, false);
}

nnml_tensor * nnml_mul_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b) {
    return nnml_mul_impl(mem, tensor_type, buffer_id, dual_idx, a, b, true);
}

// nnml_div

static nnml_tensor * nnml_div_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, bool inplace) {
    NNML_ASSERT(can_repeat(b, a));
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_DIV);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_div(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b) {
    return nnml_div_impl(mem, tensor_type, buffer_id, dual_idx, a, b, false);
}

nnml_tensor * nnml_div_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b) {
    return nnml_div_impl(mem, tensor_type, buffer_id, dual_idx, a, b, true);
}

// nnml_sqr

static nnml_tensor * nnml_sqr_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_SQR);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_sqr(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_sqr_impl(mem, tensor_type, buffer_id, dual_idx, a, false);
}

nnml_tensor * nnml_sqr_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_sqr_impl(mem, tensor_type, buffer_id, dual_idx, a, true);
}

// nnml_sqrt

static nnml_tensor * nnml_sqrt_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_SQRT);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_sqrt(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_sqrt_impl(mem, tensor_type, buffer_id, dual_idx, a, false);
}

nnml_tensor * nnml_sqrt_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_sqrt_impl(mem, tensor_type, buffer_id, dual_idx, a, true);
}

// nnml_log

static nnml_tensor * nnml_log_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_operation(NNML_OP_LOG);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_log(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_log_impl(mem, tensor_type, buffer_id, dual_idx, a, false);
}

nnml_tensor * nnml_log_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_log_impl(mem, tensor_type, buffer_id, dual_idx, a, true);
}

// nnml_mul_mat

bool nnml_can_mul_mat(const nnml_tensor * t0, const nnml_tensor * t1) {
    static_assert(NNML_MAX_DIMS == 4, "NNML_MAX_DIMS is not 4 - update this function");
    return (t0->get_elements(0) == t1->get_elements(0))     &&
           (t1->get_elements(2) % t0->get_elements(2) == 0) && // verify t0 is broadcastable
           (t1->get_elements(3) % t0->get_elements(3) == 0);
}

nnml_tensor * nnml_mul_mat(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b) {
    NNML_ASSERT(nnml_can_mul_mat(a, b));
    NNML_ASSERT(!a->is_transposed());
    const int64_t ne[4] = { a->get_elements(1), b->get_elements(1), b->get_elements(2), b->get_elements(3) };
    nnml_tensor * result = tensor_new(mem, tensor_type, buffer_id, dual_idx, NNML_TYPE_F32, 4, ne);
    result->set_operation(NNML_OP_MUL_MAT);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

void nnml_mul_mat_set_prec(nnml_tensor * a, nnml_prec prec) {
    NNML_ASSERT(a->get_operation() == NNML_OP_MUL_MAT);
    const int32_t prec_i32 = (int32_t) prec;
    a->set_op_params_i32(0, prec_i32);
}

// nnml_scale

static nnml_tensor * nnml_scale_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float s, float b, bool inplace) {
    NNML_ASSERT(a->is_padded_1d());

    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);

    float params[2] = { s, b };
    result->set_op_params(&params, sizeof(params));
    result->set_operation(NNML_OP_SCALE);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_scale(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float s) {
    return nnml_scale_impl(mem, tensor_type, buffer_id, dual_idx, a, s, 0.0, false);
}

nnml_tensor * nnml_scale_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float s) {
    return nnml_scale_impl(mem, tensor_type, buffer_id, dual_idx, a, s, 0.0, true);
}

nnml_tensor * nnml_scale_bias(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float s, float b) {
    return nnml_scale_impl(mem, tensor_type, buffer_id, dual_idx, a, s, b, false);
}

nnml_tensor * nnml_scale_bias_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float s, float b) {
    return nnml_scale_impl(mem, tensor_type, buffer_id, dual_idx, a, s, b, true);
}

// nnml_cast

nnml_tensor * nnml_cast(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_type type) {
    nnml_tensor * result = tensor_new(mem, tensor_type, buffer_id, dual_idx, type, NNML_MAX_DIMS, a->get_ne_ptr());
    result->set_name("%s (copy)", a->get_name_cstr());
    result->set_operation(NNML_OP_CPY);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, result);
    return result;
}

// nnml_norm

static nnml_tensor * nnml_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float eps, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_op_params(&eps, sizeof(eps));
    result->set_operation(NNML_OP_NORM);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps) {
    return nnml_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, eps, false);
}

nnml_tensor * nnml_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps) {
    return nnml_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, eps, true);
}

// nnml_rms_norm

static nnml_tensor * nnml_rms_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, float eps, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_op_params(&eps, sizeof(eps));
    result->set_operation(NNML_OP_RMS_NORM);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_rms_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps) {
    return nnml_rms_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, eps, false);
}

nnml_tensor * nnml_rms_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, float eps) {
    return nnml_rms_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, eps, true);
}

// nnml_rms_norm_back

nnml_tensor * nnml_rms_norm_back(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, float eps) {
    nnml_tensor * result = tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_op_params(&eps, sizeof(eps));
    result->set_operation(NNML_OP_RMS_NORM_BACK);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

// nnml_group_norm

static nnml_tensor * nnml_group_norm_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int n_groups, float eps, bool inplace) {
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_op_params_i32(0, n_groups);
    result->set_op_params_f32(1, eps);
    result->set_operation(NNML_OP_GROUP_NORM);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_group_norm(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int n_groups, float eps) {
    return nnml_group_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, n_groups, eps, false);
}

nnml_tensor * nnml_group_norm_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int n_groups, float eps) {
    return nnml_group_norm_impl(mem, tensor_type, buffer_id, dual_idx, a, n_groups, eps, true);
}

// nnml_reshape

nnml_tensor * nnml_reshape(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    NNML_ASSERT(a->is_contiguous());
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    NNML_ASSERT(a->n_elements() == b->n_elements());
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), NNML_MAX_DIMS, b->get_ne_ptr(), a, 0);
    result->set_name("%s (reshaped)", a->get_name_cstr());
    result->set_operation(NNML_OP_RESHAPE);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_reshape_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0) {
    NNML_ASSERT(a->is_contiguous());
    NNML_ASSERT(a->n_elements() == ne0);
    const int64_t ne[1] = { ne0 };
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), 1, ne, a, 0);
    result->set_name("%s (reshaped)", a->get_name_cstr());
    result->set_operation(NNML_OP_RESHAPE);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_reshape_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1) {
    NNML_ASSERT(a->is_contiguous());
    NNML_ASSERT(a->n_elements() == ne0*ne1);
    const int64_t ne[2] = { ne0, ne1 };
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), 2, ne, a, 0);
    result->set_name("%s (reshaped)", a->get_name_cstr());
    result->set_operation(NNML_OP_RESHAPE);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_reshape_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2) {
    NNML_ASSERT(a->is_contiguous());
    NNML_ASSERT(a->n_elements() == ne0*ne1*ne2);
    const int64_t ne[3] = { ne0, ne1, ne2 };
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), 3, ne, a, 0);
    result->set_name("%s (reshaped)", a->get_name_cstr());
    result->set_operation(NNML_OP_RESHAPE);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_reshape_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    NNML_ASSERT(a->is_contiguous());
    NNML_ASSERT(a->n_elements() == ne0*ne1*ne2*ne3);
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), 4, ne, a, 0);
    result->set_name("%s (reshaped)", a->get_name_cstr());
    result->set_operation(NNML_OP_RESHAPE);
    result->set_src_tensor(0, a);
    return result;
}

// nnml_view

static nnml_tensor * nnml_view_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int n_dims, const int64_t * ne, size_t offset) {
    // printf("offset=%zu\n", offset);
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), n_dims, ne, a, offset);
    result->set_name("%s (view)", a->get_name_cstr());
    result->set_op_params(&offset, sizeof(offset));
    result->set_operation(NNML_OP_VIEW);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_view_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int64_t ne0, size_t offset) {
    nnml_tensor * result = nnml_view_impl(mem, tensor_type, buffer_id, dual_idx, a, 1, &ne0, offset);
    return result;
}

nnml_tensor * nnml_view_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
    const int64_t ne[2] = { ne0, ne1 };
    nnml_tensor * result = nnml_view_impl(mem, tensor_type, buffer_id, dual_idx, a, 2, ne, offset);
    result->set_stride_bytes(1, nb1);
    result->set_stride_bytes(2, result->get_stride_bytes(1) * ne1);
    result->set_stride_bytes(3, result->get_stride_bytes(2));
    return result;
}

nnml_tensor * nnml_view_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    nnml_tensor * result = nnml_view_impl(mem, tensor_type, buffer_id, dual_idx, a, 3, ne, offset);
    result->set_stride_bytes(1, nb1);
    result->set_stride_bytes(2, nb2);
    result->set_stride_bytes(3, result->get_stride_bytes(2) * ne2);
    return result;
}

nnml_tensor * nnml_view_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    // printf("nnml_view_4d called with offset=%zu\n", offset);
    // printf(" ne0=%lld ne1=%lld ne2=%lld ne3=%lld\n", (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
    nnml_tensor * result = nnml_view_impl(mem, tensor_type, buffer_id, dual_idx, a, 4, ne, offset);
    result->set_stride_bytes(1, nb1);
    result->set_stride_bytes(2, nb2);
    result->set_stride_bytes(3, nb3);
    return result;
}

// nnml_permute

nnml_tensor * nnml_permute(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, int axis0, int axis1, int axis2, int axis3) {
    NNML_ASSERT(axis0 >= 0 && axis0 < NNML_MAX_DIMS);
    NNML_ASSERT(axis1 >= 0 && axis1 < NNML_MAX_DIMS);
    NNML_ASSERT(axis2 >= 0 && axis2 < NNML_MAX_DIMS);
    NNML_ASSERT(axis3 >= 0 && axis3 < NNML_MAX_DIMS);

    NNML_ASSERT(axis0 != axis1);
    NNML_ASSERT(axis0 != axis2);
    NNML_ASSERT(axis0 != axis3);
    NNML_ASSERT(axis1 != axis2);
    NNML_ASSERT(axis1 != axis3);
    NNML_ASSERT(axis2 != axis3);

    nnml_tensor * result = tensor_view(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_name("%s (permuted)", a->get_name_cstr());
    int ne[NNML_MAX_DIMS];
    int nb[NNML_MAX_DIMS];
    ne[axis0] = a->get_elements(0);
    ne[axis1] = a->get_elements(1);
    ne[axis2] = a->get_elements(2);
    ne[axis3] = a->get_elements(3);

    nb[axis0] = a->get_stride_bytes(0);
    nb[axis1] = a->get_stride_bytes(1);
    nb[axis2] = a->get_stride_bytes(2);
    nb[axis3] = a->get_stride_bytes(3);

    result->set_elements(0, ne[0]);
    result->set_elements(1, ne[1]);
    result->set_elements(2, ne[2]);
    result->set_elements(3, ne[3]);

    result->set_stride_bytes(0, nb[0]);
    result->set_stride_bytes(1, nb[1]);
    result->set_stride_bytes(2, nb[2]);
    result->set_stride_bytes(3, nb[3]);

    result->set_operation(NNML_OP_PERMUTE);
    result->set_src_tensor(0, a);
    int32_t params[] = { axis0, axis1, axis2, axis3 };
    result->set_op_params(params, sizeof(params));
    return result;
}

// nnml_transpose

nnml_tensor * nnml_transpose(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    nnml_tensor * result = tensor_view(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_name("%s (transposed)", a->get_name_cstr());
    result->set_elements(0, a->get_elements(1));
    result->set_elements(1, a->get_elements(0));
    result->set_stride_bytes(0, a->get_stride_bytes(1));
    result->set_stride_bytes(1, a->get_stride_bytes(0));
    result->set_operation(NNML_OP_TRANSPOSE);
    result->set_src_tensor(0, a);
    return result;
}

// nnml_rope

static nnml_tensor * nnml_rope_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, nnml_tensor * c, int n_dims, int sections[NNML_MROPE_SECTIONS], int mode,
    int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, bool inplace) {
    // printf("nnml_rope_impl called with mode=%d\n", mode);
    NNML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");
    NNML_ASSERT(b->is_vector());
    NNML_ASSERT(b->get_data_type() == NNML_TYPE_I32);
    bool mrope_used = mode & NNML_ROPE_TYPE_MROPE;
    if (mrope_used) {
        NNML_ASSERT(a->get_elements(2) * 4 == b->get_elements(0)); // mrope expecting 4 position ids per token
    } else {
        NNML_ASSERT(a->get_elements(2) == b->get_elements(0));
    }
    if (c) {
        NNML_ASSERT(c->get_data_type() == NNML_TYPE_F32);
        NNML_ASSERT(c->get_elements(0) >= n_dims / 2);
    }
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a)
                         : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    if (mrope_used && sections) {
        memcpy(params + 11, sections,  sizeof(int32_t) * NNML_MROPE_SECTIONS);
    } else {
        memset(params + 11, 0,         sizeof(int32_t) * NNML_MROPE_SECTIONS);
    }
    result->set_op_params(params, sizeof(params));
    result->set_operation(NNML_OP_ROPE);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    result->set_src_tensor(2, c);
    return result;
}

nnml_tensor * nnml_rope(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b,
    int n_dims, int mode) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, NULL, n_dims, NULL, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

nnml_tensor * nnml_rope_multi(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int sections[NNML_MROPE_SECTIONS], int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

nnml_tensor * nnml_rope_multi_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int sections[NNML_MROPE_SECTIONS], int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

nnml_tensor * nnml_rope_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, int n_dims, int mode) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, NULL, n_dims, NULL, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

nnml_tensor * nnml_rope_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, c, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

nnml_tensor * nnml_rope_ext_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b, nnml_tensor * c,
    int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return nnml_rope_impl(
        mem, tensor_type, buffer_id, dual_idx, a, b, c, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// nnml activation functions

static nnml_tensor * nnml_glu_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx,
    nnml_tensor * a, nnml_tensor * b, nnml_glu_op op, bool swapped) {
    NNML_ASSERT(a->is_contiguous_1());
    if (b) {
        NNML_ASSERT(b->is_contiguous_1());
        NNML_ASSERT(are_same_shape(a, b));
        NNML_ASSERT(a->get_data_type() == b->get_data_type());
    }
    int64_t ne[NNML_MAX_DIMS] = { a->get_elements(0) / 2 }; for (int i = 1; i < NNML_MAX_DIMS; i++) ne[i] = a->get_elements(i);
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), NNML_MAX_DIMS, b ? a->get_ne_ptr() : ne, NULL, 0);
    result->set_op_params_i32(0, (int32_t) op);
    result->set_op_params_i32(1, (int32_t) swapped);
    result->set_operation(NNML_OP_GLU);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, b);
    return result;
}

nnml_tensor * nnml_swiglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, NULL, NNML_GLU_OP_SWIGLU, false);
}

nnml_tensor * nnml_swiglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, b, NNML_GLU_OP_SWIGLU, false);
}

nnml_tensor * nnml_geglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, NULL, NNML_GLU_OP_GEGLU, false);
}

nnml_tensor * nnml_geglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, b, NNML_GLU_OP_GEGLU, false);
}

nnml_tensor * nnml_reglu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, NULL, NNML_GLU_OP_REGLU, false);
}

nnml_tensor * nnml_reglu_split(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * b) {
    return nnml_glu_impl(mem, tensor_type, buffer_id, dual_idx, a, b, NNML_GLU_OP_REGLU, false);
}

nnml_tensor * nnml_relu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_RELU);
}

nnml_tensor * nnml_relu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary_inplace(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_RELU);
}

nnml_tensor * nnml_silu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_SILU);
}

nnml_tensor * nnml_silu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary_inplace(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_SILU);
}

nnml_tensor * nnml_gelu(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_GELU);
}

nnml_tensor * nnml_gelu_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary_inplace(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_GELU);
}

// nnml_tanh

nnml_tensor * nnml_tanh(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_TANH);
}

nnml_tensor * nnml_tanh_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_unary_inplace(mem, tensor_type, buffer_id, dual_idx, a, NNML_UNARY_OP_TANH);
}

// nnml_soft_max

static nnml_tensor * nnml_soft_max_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    nnml_tensor * mask, float scale, float max_bias, bool inplace) {
    NNML_ASSERT(a->is_contiguous());
    if (mask) {
        NNML_ASSERT(mask->get_data_type() == NNML_TYPE_F16 || mask->get_data_type() == NNML_TYPE_F32);
        NNML_ASSERT(mask->is_contiguous());
        NNML_ASSERT(mask->get_elements(0) == a->get_elements(0));
        NNML_ASSERT(mask->get_elements(1) >= a->get_elements(1));
        NNML_ASSERT(a->get_elements(2) % mask->get_elements(2) == 0);
        NNML_ASSERT(a->get_elements(3) % mask->get_elements(3) == 0);
    }
    if (max_bias > 0.0f) {
        NNML_ASSERT(mask);
    }
    nnml_tensor * result = inplace ? tensor_view(mem, tensor_type, buffer_id, dual_idx, a) : tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    float params[] = { scale, max_bias };
    result->set_op_params(params, sizeof(params));
    result->set_operation(NNML_OP_SOFT_MAX);
    result->set_src_tensor(0, a);
    result->set_src_tensor(1, mask);
    return result;
}

nnml_tensor * nnml_soft_max(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_soft_max_impl(mem, tensor_type, buffer_id, dual_idx, a, NULL, 1.0f, 0.0f, false);
}

nnml_tensor * nnml_soft_max_inplace(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_soft_max_impl(mem, tensor_type, buffer_id, dual_idx, a, NULL, 1.0f, 0.0f, true);
}

nnml_tensor * nnml_soft_max_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, nnml_tensor * mask, float scale, float max_bias) {
    return nnml_soft_max_impl(mem, tensor_type, buffer_id, dual_idx, a, mask, scale, max_bias, false);
}

void nnml_soft_max_add_sinks(nnml_tensor * a, nnml_tensor * sinks) {
    if (!sinks) {
        a->set_src_tensor(2, NULL);
        return;
    }
    NNML_ASSERT(a->get_operation() == NNML_OP_SOFT_MAX);
    NNML_ASSERT(a->get_src_tensor(2) == NULL);
    NNML_ASSERT(a->get_src_tensor(0)->get_elements(2) == sinks->get_elements(0));
    NNML_ASSERT(sinks->get_data_type() == NNML_TYPE_F32);
    a->set_src_tensor(2, sinks);
}

// nnml_cont

static nnml_tensor * nnml_cont_impl(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    nnml_tensor * result = tensor_dup(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_name("%s (cont)", a->get_name_cstr());
    result->set_operation(NNML_OP_CONT);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_cont(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    return nnml_cont_impl(mem, tensor_type, buffer_id, dual_idx, a);
}

nnml_tensor * nnml_cont_4d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    NNML_ASSERT(a->n_elements() == (ne0*ne1*ne2*ne3));

    nnml_tensor * result = tensor_new_4d(mem, tensor_type, buffer_id, dual_idx, a->get_data_type(), ne0, ne1, ne2, ne3);
    result->set_name("%s (cont)", a->get_name_cstr());
    result->set_operation(NNML_OP_CONT);
    result->set_src_tensor(0, a);
    return result;
}

nnml_tensor * nnml_cont_1d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0) {
    return nnml_cont_4d(mem, tensor_type, buffer_id, dual_idx, a, ne0, 1, 1, 1);
}

nnml_tensor * nnml_cont_2d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1) {
    return nnml_cont_4d(mem, tensor_type, buffer_id, dual_idx, a, ne0, ne1, 1, 1);
}

nnml_tensor * nnml_cont_3d(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2) {
    return nnml_cont_4d(mem, tensor_type, buffer_id, dual_idx, a, ne0, ne1, ne2, 1);
}

// scatter / gather

nnml_tensor * nnml_scatter_prepare(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    // view and OP=NNML_OP_SCATTER_PRE
    nnml_tensor * result = tensor_view(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_name("%s (pre_scatter)", a->get_name_cstr());
    result->set_operation(NNML_OP_SCATTER_PRE);
    return result;
}

nnml_tensor * nnml_scatter_copy(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * a) {
    // NNML_UNUSED(axis);
    // NNML_UNUSED(n_tensors);
    // copy and OP=NNML_OP_SCATTER
    nnml_tensor * result = tensor_view(mem, tensor_type, buffer_id, dual_idx, a);
    result->set_name("%s (scatter)", a->get_name_cstr());
    result->set_operation(NNML_OP_SCATTER);
    return result;
}

nnml_tensor * nnml_gather_copy(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor ** a, int n_tensors) {
    // copy and OP=NNML_OP_GATHER
    nnml_tensor * result = tensor_new_impl(mem, tensor_type, buffer_id, dual_idx, a[0]->get_data_type(), NNML_MAX_DIMS, a[0]->get_ne_ptr(), NULL, 0);
    result->set_name("activation (gather)");
    result->set_operation(NNML_OP_GATHER);
    for (int i = 0; i < n_tensors; i++) {
        result->set_src_tensor(i, a[i]);
    }
    result->set_op_params_i32(0, n_tensors);
    return result;
}

// nnml_flash_attn_ext

nnml_tensor * nnml_flash_attn_ext(nnml_memory_t& mem, nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, nnml_tensor * q, nnml_tensor * k, nnml_tensor * v,
    nnml_tensor * mask, float scale, float max_bias, float logit_softcap) {
    NNML_ASSERT(nnml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)
    NNML_ASSERT(q->get_elements(3) == k->get_elements(3));
    NNML_ASSERT(q->get_elements(3) == v->get_elements(3));
    if (mask) {
        NNML_ASSERT(mask->is_contiguous());
        NNML_ASSERT(mask->get_elements(1) >= NNML_PAD(q->get_elements(1), NNML_KQ_MASK_PAD) &&
                "the Flash-Attention kernel requires the mask to be padded to NNML_KQ_MASK_PAD and at least n_queries big");
        NNML_ASSERT(q->get_elements(2) % mask->get_elements(2) == 0);
        NNML_ASSERT(q->get_elements(3) % mask->get_elements(3) == 0);
    }
    if (max_bias > 0.0f) {
        NNML_ASSERT(mask);
    }
    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->get_elements(0), q->get_elements(2), q->get_elements(1), q->get_elements(3) };
    nnml_tensor * result = tensor_new(mem, tensor_type, buffer_id, dual_idx, NNML_TYPE_F32, 4, ne);
    float params[] = { scale, max_bias, logit_softcap };
    result->set_op_params(params, sizeof(params));
    result->set_operation(NNML_OP_FLASH_ATTN_EXT);
    result->set_src_tensor(0, q);
    result->set_src_tensor(1, k);
    // printf("k type = %d\n", k->get_data_type());
    result->set_src_tensor(2, v);
    result->set_src_tensor(3, mask);
    return result;
}

void nnml_flash_attn_ext_add_sinks(nnml_tensor * a, nnml_tensor * sinks) {
    if (!sinks) {
        a->set_src_tensor(4, NULL);
        return;
    }
    NNML_ASSERT(a->get_operation() == NNML_OP_FLASH_ATTN_EXT);
    NNML_ASSERT(a->get_src_tensor(4) == NULL);
    NNML_ASSERT(a->get_src_tensor(0)->get_elements(2) == sinks->get_elements(0));
    NNML_ASSERT(sinks->get_data_type() == NNML_TYPE_F32);
    a->set_src_tensor(4, sinks);
}

void nnml_flash_attn_ext_set_prec(nnml_tensor * a, nnml_prec prec) {
    NNML_ASSERT(a->get_operation() == NNML_OP_FLASH_ATTN_EXT);
    const int32_t prec_i32 = (int32_t) prec;
    a->set_op_params_i32(3, prec_i32); // scale is on first pos, max_bias on second
}
