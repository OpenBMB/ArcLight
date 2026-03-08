#include <atomic>

#include "ops.h"
#include "tensor.h"


#if !defined(NNML_CPU_FP32_TO_FP16)
#define NNML_CPU_FP32_TO_FP16(x) NNML_COMPUTE_FP32_TO_FP16(x)
#endif

nnml_fp16_t f32_to_f16(float x) {
    return NNML_CPU_FP32_TO_FP16(x);
}

void nnml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void nnml_cpu_fp32_to_fp16(const float * x, nnml_fp16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = NNML_CPU_FP32_TO_FP16(x[i]);
    }
}

void quantize_row_q4_0(const float * NNML_RESTRICT x, void * NNML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, (block_q4_0 *)y, k);
}

// gemm

static void nnml_compute_forward_mul_mat_one_chunk(
    const nnml_compute_state * params,
    nnml_tensor * node,
    const nnml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    nnml_tensor * src0 = node->get_src_tensor(0);
    nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = src1->is_contiguous();

    // if (params->ith == 0) printf("src0->datatype: %d\n", src0->get_data_type());
    nnml_vec_dot_t const vec_dot      = nnml_get_type_traits(src0->get_data_type())->vec_dot;
    nnml_type const vec_dot_type      = nnml_get_type_traits(src0->get_data_type())->vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->get_data_type() == vec_dot_type) ? src1->tensor_data() : params->work_data;
    const size_t row_size = nnml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->get_data_type() != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->tensor_data() + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->get_data_type() != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)node->tensor_data() + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void forward_mul_mat_generic(nnml_tensor * node, const nnml_compute_state * params) {
    struct nnml_tensor * src0 = node->get_src_tensor(0);
    struct nnml_tensor * src1 = node->get_src_tensor(1);

    NNML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    nnml_type           vec_dot_type         = nnml_get_type_traits(src0->get_data_type())->vec_dot_type;
    nnml_from_float_t   from_float           = nnml_get_type_traits(vec_dot_type)->from_float_cpu;
    int64_t             vec_dot_num_rows     = nnml_get_type_traits(src0->get_data_type())->nrows;

    NNML_ASSERT(ne0 == ne01);
    NNML_ASSERT(ne1 == ne11);
    NNML_ASSERT(ne2 == ne12);
    NNML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    NNML_ASSERT(nb00 == nnml_type_size(src0->get_data_type()));
    NNML_ASSERT(nb10 == nnml_type_size(src1->get_data_type()));

    // dst cannot be transposed or permuted
    NNML_ASSERT(nb0 == sizeof(float));
    NNML_ASSERT(nb0 <= nb1);
    NNML_ASSERT(nb1 <= nb2);
    NNML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
    if (src1->get_data_type() != vec_dot_type) {
        char * wdata = (char *)params->work_data;

        const size_t nbw0 = nnml_type_size(vec_dot_type);
        const size_t nbw1 = nnml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->work_size >= ne13*nbw3);
        NNML_ASSERT(src1->get_data_type() == NNML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = nnml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->tensor_data() + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadgrp->current_chunk, nth, std::memory_order_relaxed);
    }

    // nnml_barrier(params->threadgrp);
    nnml_barrier_global(params->threadgrp->parent_pool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/nnml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || params->threadgrp->affinity_mode == nnml_affinity_mode::NUMA_DISTRIBUTED) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        
        nnml_compute_forward_mul_mat_one_chunk(params, node, src0->get_data_type(), num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadgrp->current_chunk, 1, std::memory_order_relaxed);
    }
}

void nnml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef NNML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}
