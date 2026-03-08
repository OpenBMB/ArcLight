/**
 * @file scheduler.cpp
 * @brief Implementation of scheduler.h
 * 
 * This file implements the nnml_compute_node function and the nnml_scheduler class.
 */
#include "scheduler.h"
#include "thread.h"
#include "ops.h"


int32_t n_steps_global = 0;                 // for debugging


void nnml_compute_node(nnml_tensor * node, const nnml_compute_state * params) {
    NNML_ASSERT(params);
    if (node->get_operation() == NNML_OP_NONE || node->is_empty()) {
        return;
    }

    switch (node->get_operation()) {
        case NNML_OP_DUP:
            {
                nnml_compute_forward_dup(node, params);
            } break;
        case NNML_OP_ADD:
            {
                nnml_compute_forward_add(node, params);
            } break;
        case NNML_OP_MUL:
            {
                nnml_compute_forward_mul(node, params);
            } break;
        case NNML_OP_MUL_MAT:
            {
                nnml_compute_forward_mul_mat(node, params);
            } break;
        case NNML_OP_GET_ROWS:
            {
                nnml_compute_forward_get_rows(node, params);
            } break;
        case NNML_OP_SET_ROWS:
            {
                nnml_compute_forward_set_rows(node, params);
            } break;
        case NNML_OP_CPY:
            {
                nnml_compute_forward_cpy(node, params);
            } break;
        case NNML_OP_NORM:
            {
                nnml_compute_forward_norm(node, params);
            } break;
        case NNML_OP_RMS_NORM:
            {
                nnml_compute_forward_rms_norm(node, params);
            } break;
        case NNML_OP_GROUP_NORM:
            {
                nnml_compute_forward_group_norm(node, params);
            } break;
        case NNML_OP_RESHAPE:
            {
                nnml_compute_forward_reshape(node, params);
            } break;
        case NNML_OP_VIEW:
            {
                nnml_compute_forward_view(node, params);
            } break;
        case NNML_OP_PERMUTE:
            {
                nnml_compute_forward_permute(node, params);
            } break;
        case NNML_OP_TRANSPOSE:
            {
                nnml_compute_forward_transpose(node, params);
            } break;
        case NNML_OP_ROPE:
            {
                nnml_compute_forward_rope(node, params);
            } break;
        case NNML_OP_UNARY:
            {
                nnml_compute_forward_unary(node, params);
            } break;
        case NNML_OP_GLU:
            {
                nnml_compute_forward_glu(node, params);
            } break;
        case NNML_OP_SOFT_MAX:
            {
                nnml_compute_forward_soft_max(node, params);
            } break;
        case NNML_OP_CONT:
            {
                nnml_compute_forward_cont(node, params);
            } break;
        case NNML_OP_FLASH_ATTN_EXT:
            {
                nnml_compute_forward_flash_attn_ext(node, params);
            } break;
        case NNML_OP_SCATTER_PRE:
            {
                nnml_compute_forward_scatter_pre(node, const_cast<nnml_compute_state*>(params));
            } break;
        case NNML_OP_SCATTER:
            {
                // we only allow modify params inside scatter/gather function
                nnml_compute_forward_scatter(node, const_cast<nnml_compute_state*>(params));
            } break;
        case NNML_OP_GATHER:
            {
                nnml_compute_forward_gather(node, const_cast<nnml_compute_state*>(params));
            } break;
        case NNML_OP_NONE:          // nop
            break;
        case NNML_OP_COUNT:
            NNML_ABORT("fatal error");
    }
}

void nnml_single_graph_compute_thread(void * data) {
    nnml_compute_state * state = (nnml_compute_state *) data;
    
    for (int node_n = 0; node_n < single_graph->get_n_nodes();) {

        nnml_tensor * node = single_graph->get_nth_cnode(node_n)->tensor;

        if (node->get_operation() == NNML_OP_SCATTER_PRE || node->get_operation() == NNML_OP_GATHER) {
            nnml_barrier_global(threadpool);
        }

        nnml_compute_node(node, state);

        if (node_n + 1 < single_graph->get_n_nodes()) {
            if (node->get_operation() == NNML_OP_SCATTER_PRE || node->get_operation() == NNML_OP_GATHER) {
                nnml_barrier_global(threadpool);
            } else if (state->threadgrp->parent_pool->view_groups > 1) {
                // for non-scatter/gather nodes, we can use threadgroup barrier for better performance, but only when there are multiple groups
                nnml_barrier(state->threadgrp);
            } else nnml_barrier_global(threadpool);
        }
        node_n = single_graph->get_nth_cnode(node_n)->child[state->threadgrp->crt_group_id];
    }
    nnml_barrier_global(threadpool);
}

void nnml_scheduler::init(nnml_threadpool * tp, std::vector<nnml_cgraph *> & graphs) {
    NNML_ASSERT(tp);
    NNML_ASSERT(tp->total_threads > 0);
    NNML_ASSERT(tp->total_groups > 0);
    for (int i = 0; i < tp->total_groups; i++) {
        nnml_threadgroup * group = tp->get_group(i);
        NNML_ASSERT(group);
        NNML_ASSERT(group->work_fn == nnml_single_graph_compute_thread);
    }
    threadpool = tp;

    if (graphs.size() == 1) {
        single_graph = graphs[0];
    } else {
        multi_graphs = std::move(graphs);
    }
}

nnml_status nnml_scheduler::nnml_single_graph_compute() {
    NNML_ASSERT(single_graph);
#ifdef NNML_USE_OPENMP
    // not implemented in this version
#else
    nnml_threadpool_kickoff(threadpool, threadpool->total_threads/threadpool->total_groups);
    nnml_threadgroup_run_worker0(threadpool->get_group(0));
#endif
    n_steps_global++;
    return NNML_STATUS_SUCCESS;
}

