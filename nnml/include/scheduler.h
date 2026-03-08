/**
 * @file scheduler.h
 * @brief computation scheduler for NNML.
 * 
 * Provides:
 *  - Single graph computation with sub-graph support (e.g. cross-NUMA Tensor-parallelism)
 *  - Multi-graph computation with inter-graph synchronization (e.g. pipeline parallelism)
 *  - Graph node compute dispatching to operator library
 */
#pragma once

#include <stdlib.h>

#include "tensor.h"
#include "thread.h"
#include "cgraph.h"


extern int32_t n_steps_global;                 // for debugging

nnml_threadpool * threadpool   = nullptr;
nnml_cgraph *     single_graph = nullptr;
std::vector<nnml_cgraph *> multi_graphs;


/**
 * nnml_scheduler: the main scheduler class that start computation across single or multiple graphs
 *  - Init with thread pool and graphs
 *  - Start single graph computation (including multi-subgraph support)
 *  - Start multi-graph computation (e.g. pipeline parallelism)
 */
class nnml_scheduler {
public:
    nnml_scheduler() = default;
    ~nnml_scheduler() = default;

    void init(nnml_threadpool * tp, std::vector<nnml_cgraph *> & graphs);     // initialize and prepare thread group

    nnml_status nnml_single_graph_compute();   // based on single graph or composite graph with sub-graphs, 
                                               // and split-merge thread groups, kickoff single graph computation
    nnml_status nnml_multi_graph_compute();    // based on multiple graphs and thread groups, kickoff multi-graph computation, handle inter-graph synchronization
};

/**
 * nnml_single_graph_compute_thread: entry func of each thread for graph computation, it call nnml_compute_node sequentially
 * nnml_compute_node: entry func for each graph node computation, it dispatches the actual computation to operator library based on node type and parameters
 */
void nnml_single_graph_compute_thread(void * data);                                // single-threaded graph computation in topological order, handle intra-node and sub-graph synchronization
void nnml_compute_node(nnml_tensor * node, const nnml_compute_state * params);     // graph node compute entry, dispatch to operator library
