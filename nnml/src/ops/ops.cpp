#include <cassert>
#include <algorithm>
#include <float.h>

#include "ops.h"
#include "tensor.h"
#include "unistd.h"


// branches of get_rows

static void nnml_compute_forward_get_rows_q(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = src1->n_elements();

    const nnml_type type = src0->get_data_type();
    nnml_to_float_t const dequantize_row_q = nnml_get_type_traits(type)->to_float;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == nnml_type_size(type));
    assert(node->n_rows() == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->tensor_data() + i10*nb10 + i11*nb11 + i12*nb12);

        NNML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (const void *) ((char *)  src0->tensor_data() + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  node->tensor_data() + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void nnml_compute_forward_get_rows_f16(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = src1->n_elements();

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(nnml_fp16_t));
    assert(node->n_rows() == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->tensor_data() + i10*nb10 + i11*nb11 + i12*nb12);

        NNML_ASSERT(i01 >= 0 && i01 < ne01);

        nnml_cpu_fp16_to_fp32(
            (const nnml_fp16_t*) ((char *) src0->tensor_data() + i01*nb01 + i11*nb02 + i12*nb03),
                       (float *) ((char *)  node->tensor_data() + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void nnml_compute_forward_get_rows_f32(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = src1->n_elements();

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(float));
    assert(node->n_rows() == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    // if (params->ith == 0) node->get_src_tensor(0)->print_data(2560*4, true);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->tensor_data() + i10*nb10 + i11*nb11 + i12*nb12);

        NNML_ASSERT(i01 >= 0 && i01 < ne01);

        nnml_vec_cpy_f32(nc,
                (float *) ((char *) node->tensor_data() + i10*nb1  + i11*nb2  + i12*nb3),
                (float *) ((char *) src0->tensor_data() + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

// branches of set_rows

template<typename idx_t>
static void nnml_compute_forward_set_rows_f32(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ne01;

    assert(ne0  == nc);
    assert(ne2  == ne02);
    assert(ne3  == ne03);
    assert(src0->get_data_type() == NNML_TYPE_F32);
    assert(ne02 % ne11 == 0);
    assert(ne03 % ne12 == 0);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = std::min(ir0 + dr, nr);

    nnml_from_float_t const from_float = nnml_get_type_traits(node->get_data_type())->from_float_cpu;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i = ir0; i < ir1; ++i) {
                const int64_t i12 = i03%ne12;
                const int64_t i11 = i02%ne11;
                const int64_t i10 = i;

                const int64_t i1 = *(idx_t *) ((char *) src1->tensor_data() + i10*nb10 + i11*nb11 + i12*nb12);

                NNML_ASSERT(i1 >= 0 && i1 < ne1);

                from_float(
                        (const float *) ((char *) src0->tensor_data() +  i*nb01 + i02*nb02 + i03*nb03),
                                        ((char *) node->tensor_data() + i1*nb1  + i02*nb2  + i03*nb3), nc);
            }
        }
    }
}

// branches of dup

static void nnml_compute_forward_dup_same_cont(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);

    NNML_ASSERT(node->n_elements() == src0->n_elements());
    NNML_ASSERT(node->is_contiguous() && src0->is_contiguous());
    NNML_ASSERT(src0->get_data_type() == node->get_data_type());

    const size_t nb0 = nnml_type_size(src0->get_data_type());

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by blocks
    const int nk = src0->n_elements()/nnml_blck_size(src0->get_data_type());
    const int dr = (nk + nth - 1) / nth;
    const int k0 = dr * ith;
    const int k1 = MIN(k0 + dr, nk);

    if (k0 < k1) {
        memcpy(
            ((char *) node->tensor_data() + k0*nb0),
            ((char *) src0->tensor_data() + k0*nb0),
            (k1 - k0) * nb0);
    }
}

static void nnml_compute_forward_dup_bytes(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);

    NNML_ASSERT(node->n_elements() == src0->n_elements());
    NNML_ASSERT(src0->get_data_type() == node->get_data_type());

    NNML_TENSOR_UNARY_OP_LOCALS;

    if (src0->is_contiguous() && node->is_contiguous()) {
        nnml_compute_forward_dup_same_cont(node, params);
        return;
    }

    const size_t type_size = nnml_type_size(src0->get_data_type());

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->get_data_type() == node->get_data_type() &&
        are_same_shape(src0, node) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        const size_t rs = nnml_row_size(src0->get_data_type(), ne00);
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *) node->tensor_data() + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (node->is_contiguous()) {
        size_t id = 0;
        char * dst_ptr = (char *)node->tensor_data();
        const size_t rs = ne00 * type_size;

        if (nb00 == type_size) {
            // src0 is contigous on first dimension, copy by rows
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        const char * src0_ptr = (char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } else {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = (char *) src0->tensor_data() + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        return;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    const int64_t nk00 = ne00 / nnml_blck_size(src0->get_data_type());
    const int64_t nk0  = ne0  / nnml_blck_size(node->get_data_type());

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t k00 = 0; k00 < nk00; k00++) {
                    const char * src0_ptr = ((char *) src0->tensor_data() + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *) node->tensor_data() + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    if (++k10 == nk0) {
                        k10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

template<typename src_t, typename dst_t>
static void nnml_compute_forward_dup_flt(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);

    NNML_ASSERT(node->n_elements() == src0->n_elements());
    NNML_ASSERT(!nnml_is_quantized(src0->get_data_type()) && !nnml_is_quantized(node->get_data_type()));

    NNML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    if (src0->get_data_type() == node->get_data_type() &&
        ne00 == ne0 &&
        nb00 == nnml_type_size(src0->get_data_type()) && nb0 == nnml_type_size(node->get_data_type())) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *) node->tensor_data() + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // case: dst tensor is contiguous
    if (node->is_contiguous()) {
        if (nb00 == sizeof(src_t)) {
            if constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) node->tensor_data();

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) node->tensor_data();

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const src_t * src0_ptr = (src_t *) ((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                float tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) node->tensor_data();

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    for (int i01 = ir0; i01 < ir1; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const src_t * src0_ptr = (src_t *) ((char *) src0->tensor_data() + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            float tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if constexpr (std::is_same_v<dst_t, src_t>) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->tensor_data() + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  node->tensor_data() + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(dst_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } else {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->tensor_data() + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *) node->tensor_data() + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        float tmp = type_conversion_table<src_t>::to_f32(*(const src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename src_t>
static void nnml_compute_forward_dup_to_q(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);

    NNML_ASSERT(node->n_elements() == src0->n_elements());
    NNML_ASSERT(!nnml_is_quantized(src0->get_data_type()));

    NNML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (node->is_contiguous() &&
            nb00 == sizeof(src_t) &&
            nnml_get_type_traits(node->get_data_type())->from_float_cpu) {
        // casting non-quantized types --> intermediate f32 --> quantized
        nnml_from_float_t const quantize_row_q = nnml_get_type_traits(node->get_data_type())->from_float_cpu;
        float * src0_f32 = (float *) params->work_data + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / nnml_blck_size(node->get_data_type()));
        char * dst_ptr = (char *) node->tensor_data();

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                for (int i01 = ir0; i01 < ir1; i01++) {
                    const src_t * src0_ptr = (src_t *) ((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03);

                    for (int i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } else {
        // printf("%s %s\n", nnml_type_name(src0->type), nnml_type_name(dst->type));
        NNML_ABORT("not implemented");
    }
}

static void nnml_compute_forward_dup_from_q(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const nnml_type type = src0->get_data_type();
    nnml_to_float_t const dequantize_row_q = nnml_get_type_traits(type)->to_float;

    size_t qk = nnml_blck_size(type);
    const int64_t nr = src1->n_elements() / qk;

    // destination must be contiguous in the first dimension
    NNML_ASSERT(nb10 == nnml_type_size(node->get_data_type()));
    // must either have first dimension large enough to hold a row, or fully contiguous
    NNML_ASSERT((ne10 % qk) == 0 || node->is_contiguous());

    const int ith = params->ith;
    const int nth = params->nth;

    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        const int64_t i03 = i/(ne00 * ne01 * ne02);
        const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        const int64_t i13 = i/(ne10 * ne11 * ne12);
        const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (const void *) ((char *) src0->tensor_data() + x_offset),
                     (float *) ((char *) node->tensor_data() + dst_offset), qk);
    }
}

// branches of rms

static void nnml_compute_forward_rms_norm_f32(nnml_tensor * node, const nnml_compute_state * params) {
    // printf("node name: %s\n", node->get_name_cstr());
    const nnml_tensor * src0 = node->get_src_tensor(0);

    NNML_ASSERT(are_same_shape(src0, node));
    NNML_ASSERT(src0->get_stride_bytes(0) == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    NNML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, node->get_operation_params(), sizeof(float));

    NNML_ASSERT(eps >= 0.0f);
    // printf("eps: %f\n", eps);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->tensor_data() + i01*nb01 + i02*nb02 + i03*nb03);
                nnml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    // if (std::isnan(x[i00])) {
                    //     printf("nan detected at i03: %ld i02: %ld i01: %ld i00: %ld\n", i03, i02, i01, i00);
                    // }
                    sum += (nnml_float)(x[i00] * x[i00]);
                }
                const float mean = sum/ne00;
                float * y = (float *) ((char *) node->tensor_data() + i01*nb1 + i02*nb2 + i03*nb3);
                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }
                // printf("mean: %f\n", mean);     // nan detected here
                const float scale = 1.0f/sqrtf(mean + eps);
                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);
                nnml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

// branches of rope

static float nnml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void nnml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(nnml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(nnml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

static void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

void nnml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

void nnml_mrope_cache_init(
     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;  // extra position id for vision encoder
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    NNML_ASSERT(sect_dims <= ne0);

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        }
        else if (sector >= sec_w && sector < sec_w + sections[2]) {
            theta = theta_w;
        }
        else if (sector >= sec_w + sections[2]) {
            theta = theta_e;
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

static void nnml_compute_forward_rope_f32(nnml_tensor * node, const nnml_compute_state * params) {

    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    const nnml_tensor * src2 = node->get_src_tensor(2);
    
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];

    //const int n_past     = ((int32_t *) node->get_operation_params())[0];
    const int n_dims     = ((int32_t *) node->get_operation_params())[1];
    const int mode       = ((int32_t *) node->get_operation_params())[2];
    //const int n_ctx      = ((int32_t *) node->get_operation_params())[3];
    const int n_ctx_orig = ((int32_t *) node->get_operation_params())[4];

    memcpy(&freq_base,   (int32_t *) node->get_operation_params() +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) node->get_operation_params() +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) node->get_operation_params() +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) node->get_operation_params() +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) node->get_operation_params() +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) node->get_operation_params() + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) node->get_operation_params() + 11, sizeof(int)*4);

    NNML_TENSOR_UNARY_OP_LOCALS

    NNML_ASSERT(nb00 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = node->n_rows();

    NNML_ASSERT(n_dims <= ne0);
    NNML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    nnml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    
    const bool is_neox = mode & NNML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & NNML_ROPE_TYPE_MROPE;  // nnml_rope_multi, multimodal rotary position embedding
    
    if (is_mrope) {
        NNML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        NNML_ASSERT(src2->get_data_type() == NNML_TYPE_F32);
        NNML_ASSERT(src2->get_elements(0) >= n_dims / 2);
        freq_factors = (const float *) src2->tensor_data();
    }

    const float sin_sign = 1.0f;
    const int32_t * pos = (const int32_t *) src1->tensor_data();

    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len

            float * cache = (float *) params->work_data + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            if (!is_mrope) {
                const int64_t p = pos[i2];
                nnml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                nnml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, false, freq_scale, freq_factors, corr_dims, ne0, 
                                      ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            
            for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (is_neox || is_mrope) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        // printf(".");
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_data  = (float *)((char *)  node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims/2];

                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *) node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    }
                }
                // fill the remain channels with data from src tensor
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const float * const src = (float *)((char *) src0->tensor_data() + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    float * dst_data  = (float *)((char *)  node->tensor_data() + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

// branches of add

static void nnml_compute_forward_add_q_f32(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    NNML_ASSERT(are_same_shape(src0, src1) && are_same_shape(src0, node));
    const int nr  = src0->n_rows();

    NNML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const nnml_type type = src0->get_data_type();
    const nnml_type dtype = node->get_data_type();
    nnml_to_float_t const dequantize_row_q = nnml_get_type_traits(type)->to_float;
    nnml_from_float_t const quantize_row_q = nnml_get_type_traits(dtype)->from_float_cpu;

    // we don't support permuted src0 or src1
    NNML_ASSERT(nb00 == nnml_type_size(type));
    NNML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    NNML_ASSERT(nb0 <= nb1);
    NNML_ASSERT(nb1 <= nb2);
    NNML_ASSERT(nb2 <= nb3);

    NNML_ASSERT(nnml_is_quantized(src0->get_data_type()));
    NNML_ASSERT(src1->get_data_type() == NNML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->work_data + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        const int i13 = i03;
        const int i12 = i02;
        const int i11 = i01;

        const int i3 = i03;
        const int i2 = i02;
        const int i1 = i01;

        void  * src0_row = (void *) ((char *) src0->tensor_data() + (i01*nb01 + i02*nb02 + i03*nb03));
        float * src1_row = (float *)((char *) src1->tensor_data() + (i11*nb11 + i12*nb12 + i13*nb13));
        void  * dst_row  = (void *) ((char *) node->tensor_data() + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        assert(ne00 % 32 == 0);
        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        nnml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        if (quantize_row_q != NULL) {
            quantize_row_q(wdata, dst_row, ne00);
        } else {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

// branches of glu


static void nnml_compute_forward_swiglu_f32(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    char * src0_d = (char *) src0->tensor_data();
    char * src1_d = (char *) (src1 ? src1->tensor_data() : src0->tensor_data());
    const size_t src0_o = src0->get_stride_bytes(1);
    const size_t src1_o = src1 ? src1->get_stride_bytes(1) : src0->get_stride_bytes(1);

    NNML_ASSERT(src0->is_contiguous_1());
    NNML_ASSERT(node->is_contiguous_1());

    if (src1) {
        NNML_ASSERT(src1->is_contiguous_1());
        NNML_ASSERT(src0->get_data_type() == src1->get_data_type());
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->get_elements(0) : src0->get_elements(0) / 2;
    const int nr = src0->n_rows();

    NNML_ASSERT(node->get_elements(0) == nc);
    NNML_ASSERT(node->n_rows() == nr);

    const int32_t swapped = node->get_op_params_i32(1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        nnml_vec_swiglu_f32(nc, (float *) ((char *) node->tensor_data() + i1*(node->get_stride_bytes(1))), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) node->tensor_data() + i1*( node->get_stride_bytes(1))))[k];
            NNML_UNUSED(x);
            assert(!std::isnan(x));
            assert(!std::isinf(x));
        }
#endif
    }
}

static void nnml_compute_forward_swiglu_f16(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);
    char * src0_d = (char *) src0->tensor_data();
    char * src1_d = (char *) (src1 ? src1->tensor_data() : src0->tensor_data());
    const size_t src0_o = src0->get_stride_bytes(1);
    const size_t src1_o = src1 ? src1->get_stride_bytes(1) : src0->get_stride_bytes(1);

    NNML_ASSERT(src0->is_contiguous_1());
    NNML_ASSERT(node->is_contiguous_1());

    if (src1) {
        NNML_ASSERT(src1->is_contiguous_1());
        NNML_ASSERT(src0->get_data_type() == src1->get_data_type());
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->get_elements(0) : src0->get_elements(0) / 2;
    const int nr = src0->n_rows();

    NNML_ASSERT(node->get_elements(0) == nc);
    NNML_ASSERT(node->n_rows() == nr);

    const int32_t swapped = node->get_op_params_i32(1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        nnml_fp16_t * src0_p = (nnml_fp16_t *) (src0_d + i1*src0_o);
        nnml_fp16_t * src1_p = (nnml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        nnml_vec_swiglu_f16(nc, (nnml_fp16_t *) ((char *) node->tensor_data() + i1*(node->get_stride_bytes(1))), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const nnml_fp16_t x = ((nnml_fp16_t *) ((char *) node->tensor_data() + i1*( node->get_stride_bytes(1))))[k];
            const float v = NNML_FP16_TO_FP32(x);
            NNML_UNUSED(v);
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
#endif
    }
}

static void nnml_compute_forward_swiglu(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);

    switch (src0->get_data_type()) {
        case NNML_TYPE_F32:
            {
                nnml_compute_forward_swiglu_f32(node, params);
            } break;
        case NNML_TYPE_F16:
            {
                abort();
                // nnml_compute_forward_swiglu_f16(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}


std::pair<int64_t, int64_t> get_thread_range(nnml_tensor * node, const nnml_compute_state * params) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;
    const int64_t nr  = node->n_rows();
    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;
    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);
    return {ir0, ir1};
}

// entry of operations

void nnml_compute_forward_dup(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);

    if (src0->get_data_type() == node->get_data_type()) {
        nnml_compute_forward_dup_bytes(node, params);
        return;
    }

    switch (src0->get_data_type()) {
        case NNML_TYPE_F16:
            {
                /**/ if (node->get_data_type() == NNML_TYPE_F16)  nnml_compute_forward_dup_flt<nnml_fp16_t, nnml_fp16_t>(node, params);
                // else if (node->get_data_type() == NNML_TYPE_BF16) nnml_compute_forward_dup_flt<nnml_fp16_t, nnml_bf16_t>(params, node);
                else if (node->get_data_type() == NNML_TYPE_F32)  nnml_compute_forward_dup_flt<nnml_fp16_t, float      >(node, params);
                else nnml_compute_forward_dup_to_q<nnml_fp16_t>(node, params);
            } break;
        case NNML_TYPE_F32:
            {
                /**/ if (node->get_data_type() == NNML_TYPE_F16)  nnml_compute_forward_dup_flt<float, nnml_fp16_t>(node, params);
                // else if (node->type == NNML_TYPE_BF16) nnml_compute_forward_dup_flt<float, nnml_bf16_t>(params, node);
                else if (node->get_data_type() == NNML_TYPE_F32)  nnml_compute_forward_dup_flt<float, float      >(node, params);
                else if (node->get_data_type() == NNML_TYPE_I32)  nnml_compute_forward_dup_flt<float, int32_t    >(node, params);
                else nnml_compute_forward_dup_to_q<float>(node, params);
            } break;
        case NNML_TYPE_I32:
            {
                if (node->get_data_type() == NNML_TYPE_F32) nnml_compute_forward_dup_flt<int32_t, float>(node, params);
                else NNML_ABORT("not implemented");
            } break;
        default:
            {
                if (nnml_is_quantized(src0->get_data_type()) && node->get_data_type() == NNML_TYPE_F32) {
                    nnml_compute_forward_dup_from_q(node, params);
                    break;
                }
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_add(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    switch (src0->get_data_type()) {
        case NNML_TYPE_F32:
        case NNML_TYPE_F16:
        // case NNML_TYPE_BF16:
            {
                nnml_binary_compute_forward_add_non_quantized(node, params);
            } break;
        case NNML_TYPE_Q4_0:
        case NNML_TYPE_Q8_0:
        case NNML_TYPE_Q6_K:
            {
                nnml_compute_forward_add_q_f32(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_mul(nnml_tensor * node, const nnml_compute_state * params) {
    nnml_binary_compute_forward_mul(node, params);
}

void nnml_compute_forward_mul_mat(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    if (node->is_asm_gemm() && src0->get_data_type() == NNML_TYPE_Q4_0) {
        // printf("gemm q4_0 x q8_0 using asm\n");
        if (nnml_cpu_has_avx2() || (nnml_cpu_has_sve() && nnml_cpu_has_matmul_int8() && nnml_cpu_get_sve_cnt() == QK8_0)) {
            if (src0->get_elements(1) % 8 == 0) {
                forward_mul_mat<block_q4_0, 8, 8, NNML_TYPE_Q8_0>(node, params);
            }
        }
        if (nnml_cpu_has_neon() && nnml_cpu_has_matmul_int8()) {
            if (src0->get_elements(1) % 4 == 0) {
                forward_mul_mat<block_q4_0, 8, 4, NNML_TYPE_Q8_0>(node, params);
            }
        }
        if (nnml_cpu_has_neon() && nnml_cpu_has_dotprod()) {
            if (src0->get_elements(1) % 4 == 0) {
                forward_mul_mat<block_q4_0, 4, 4, NNML_TYPE_Q8_0>(node, params);
            }
        }
    } else {
        forward_mul_mat_generic(node, params);
    }
}

void nnml_compute_forward_get_rows(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    switch (src0->get_data_type()) {
        case NNML_TYPE_Q4_0:
        case NNML_TYPE_Q6_K:
            {
                nnml_compute_forward_get_rows_q(node, params);
            } break;
        case NNML_TYPE_F16:
            {
                nnml_compute_forward_get_rows_f16(node, params);
            } break;
        case NNML_TYPE_F32:
        case NNML_TYPE_I32:
            {
                nnml_compute_forward_get_rows_f32(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_set_rows(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    const nnml_tensor * src1 = node->get_src_tensor(1);

    switch (src0->get_data_type()) {
        case NNML_TYPE_F32:
            {
                if (src1->get_data_type() == NNML_TYPE_I64) {
                    nnml_compute_forward_set_rows_f32<int64_t>(node, params);
                } else if (src1->get_data_type() == NNML_TYPE_I32) {
                    nnml_compute_forward_set_rows_f32<int32_t>(node, params);
                } else {
                    NNML_ABORT("src1->type = %d (%s) not supported", src1->get_data_type(), nnml_type_name(src1->get_data_type()));
                }
            } break;
        default:
            {
                NNML_ABORT("src0->type = %d (%s) not supported", src0->get_data_type(), nnml_type_name(src0->get_data_type()));
            }
    }
}

void nnml_compute_forward_cpy(nnml_tensor * node, const nnml_compute_state * params) {
    nnml_compute_forward_dup(node, params);
}

void nnml_compute_forward_norm(nnml_tensor * node, const nnml_compute_state * params) {
    assert(0);
}

void nnml_compute_forward_rms_norm(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    switch (src0->get_data_type()) {
        case NNML_TYPE_F32:
            {
                nnml_compute_forward_rms_norm_f32(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_group_norm(nnml_tensor * node, const nnml_compute_state * params) {
    assert(0);
}

void nnml_compute_forward_reshape(nnml_tensor * node, const nnml_compute_state * params) {
    NNML_UNUSED(node);
    NNML_UNUSED(params);
}

void nnml_compute_forward_view(nnml_tensor * node, const nnml_compute_state * params) {
    assert(0);
}

void nnml_compute_forward_permute(nnml_tensor * node, const nnml_compute_state * params) {
    // NOP
    NNML_UNUSED(node);
    NNML_UNUSED(params);
}

void nnml_compute_forward_transpose(nnml_tensor * node, const nnml_compute_state * params) {
    assert(0);
}

void nnml_compute_forward_rope(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);

    switch (src0->get_data_type()) {
        case NNML_TYPE_F16:
            {
                nnml_compute_forward_rope_f16(node, params);
            } break;
        case NNML_TYPE_F32:
            {
                nnml_compute_forward_rope_f32(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_unary(nnml_tensor * node, const nnml_compute_state * params) {
    assert(0);
}

void nnml_compute_forward_glu(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_glu_op op = node->get_glu_op();
    switch (op) {
        case NNML_GLU_OP_REGLU:
            {
                assert(0);
                // nnml_compute_forward_reglu(node, params);
            } break;
        case NNML_GLU_OP_GEGLU:
            {
                assert(0);
                // nnml_compute_forward_geglu(node, params);
            } break;
        case NNML_GLU_OP_SWIGLU:
            {
                nnml_compute_forward_swiglu(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_soft_max(nnml_tensor * node, const nnml_compute_state * params) {
    const nnml_tensor * src0 = node->get_src_tensor(0);
    switch (src0->get_data_type()) {
        case NNML_TYPE_F32:
            {
                nnml_compute_forward_soft_max_f32(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_cont(nnml_tensor * node, const nnml_compute_state * params) {
    nnml_compute_forward_dup(node, params);
}

void nnml_compute_forward_flash_attn_ext(nnml_tensor * node, const nnml_compute_state * params) {
    switch (node->get_operation_params()[3]) {
        case NNML_PREC_DEFAULT:
        case NNML_PREC_F32:
            {
                // uses F32 accumulators
                nnml_compute_forward_flash_attn_ext_f16(node, params);
            } break;
        default:
            {
                NNML_ABORT("fatal error");
            }
    }
}

void nnml_compute_forward_scatter_pre(nnml_tensor * node, nnml_compute_state * params) {
    int n_threads_pergroup = params->nth_global / params->threadgrp->n_groups;
    
    if (params->ith % n_threads_pergroup == 0) params->threadgrp->crt_group_id = params->ith/n_threads_pergroup;
    if (params->ith == 0) params->threadgrp->parent_pool->view_groups = params->threadgrp->n_groups;
    
    params->nth = params->nth_global / params->threadgrp->n_groups;
    params->ith = params->ith_global % params->nth;
    params->work_data = params->threadgrp->work_buffer;
    
    return;
}

void nnml_compute_forward_scatter(nnml_tensor * node, nnml_compute_state * params) {
    // here we assume that i-th = ith_global
    return;
}

void nnml_compute_forward_gather(nnml_tensor * node, nnml_compute_state * params) {
    // first reorganize the threadpool
    if (params->ith == 0) params->threadgrp->crt_group_id = 0;
    if (params->ith == 0) params->threadgrp->parent_pool->view_groups = 1;
    params->nth = params->nth_global;
    params->ith = params->ith_global;
    params->work_data = params->threadgrp->parent_pool->main_buffer;
    nnml_barrier_global(params->threadgrp->parent_pool);

    // then distributed compute the gather operation
    const int n_srcs = node->get_op_params_i32(0);
    const size_t n_elements = node->n_elements();
    void * dst_data = node->tensor_data();
    nnml_type type = node->get_data_type();

    const int64_t v_size = 8;
    int64_t n_blocks = (n_elements + v_size - 1) / v_size;
    int64_t b_per_thread = n_blocks / params->nth;
    int64_t b_reminder   = n_blocks % params->nth;
    int64_t ib0 = b_per_thread * params->ith + MIN(b_reminder, params->ith);
    int64_t ib1 = b_per_thread * (params->ith + 1) + MIN(b_reminder, params->ith + 1);
    int64_t ir0 = ib0 * v_size;
    int64_t ir1 = MIN(ib1 * v_size, n_elements);

    const size_t thread_n = ir1 - ir0;
    if (thread_n <= 0) NNML_ABORT("fatal error: thread_n <= 0.");

    if (type == NNML_TYPE_F32) {
        const float * local_srcs[n_srcs];
        for (int i = 0; i < n_srcs; ++i) {
            local_srcs[i] = (const float *)node->get_src_tensor(i)->tensor_data() + ir0;
        }
        float * local_dst = (float *)dst_data + ir0;
        nnml_vec_add_f32((float*)local_dst, (const float**)local_srcs, n_srcs, thread_n);
    } else if (type == NNML_TYPE_F16) {
        const __fp16 * local_srcs[n_srcs];
        for (int i = 0; i < n_srcs; ++i) {
            local_srcs[i] = (const __fp16 *)node->get_src_tensor(i)->tensor_data() + ir0;
        }
        __fp16 * local_dst = (__fp16 *)dst_data + ir0;
        nnml_vec_add_f16((__fp16*)local_dst, (const __fp16**)local_srcs, n_srcs, thread_n);
    } else {
        NNML_ABORT("fatal error: unsupported data type %d (%s) in gather operation.", type, nnml_type_name(type));
    }

    return;
}
