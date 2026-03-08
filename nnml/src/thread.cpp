/**
 * @file thread.cpp
 * @brief Implementation of thread.h
 * 
 * This file implements the nnml_threadgroup and nnml_threadpool classes, which manage worker threads for parallel computation.
 * The implementation supports both standard threads and OpenMP (reserved for future), with NUMA-aware affinity management.
 * The thread pool can be configured with multiple groups and threads per group, and provides APIs for synchronization and work buffer management.
 */
#include <cerrno>
#include <thread>
#include <unistd.h>
#include <map>

#if defined(NNML_USE_OPENMP)
#include <omp.h>
#endif

#include "thread.h"


int boot_node_id = -1;

#if defined(_WIN32) && defined(_MSC_VER) && !defined(__clang__)
void atomic_store(atomic_int *ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
void atomic_store_explicit(atomic_int *ptr, LONG val, memory_order) {
    InterlockedExchange(ptr, val);
}
LONG atomic_load(atomic_int *ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
LONG atomic_load_explicit(atomic_int *ptr, memory_order) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
LONG atomic_fetch_add(atomic_int *ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
LONG atomic_fetch_add_explicit(atomic_int *ptr, LONG inc, memory_order) {
    return InterlockedExchangeAdd(ptr, inc);
}
atomic_bool atomic_flag_test_and_set(atomic_flag *ptr) {
    return InterlockedExchange(ptr, 1);
}
void atomic_flag_clear(atomic_flag *ptr) {
    InterlockedExchange(ptr, 0);
}
void atomic_thread_fence(memory_order) {
    MemoryBarrier();
}
#endif


// implementation of nnml_threadgroup
void nnml_threadgroup::init(int n_threads, int group_id, int total_groups, nnml_threadpool * parent_pool) {
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&cond, nullptr);
    n_threads_max = n_threads;
    crt_group_id = group_id;
    n_groups = total_groups;
    workers = new nnml_compute_state[n_threads];
    for (int i = 0; i < n_threads; i++) {
        workers[i].ith = i;
        workers[i].nth = n_threads;
        workers[i].threadgrp = this;
        workers[i].ith_global = group_id * n_threads + i;
        workers[i].nth_global = total_groups * n_threads;
        workers[i].cpu_core = -1;
#ifndef NNML_USE_OPENMP
        workers[i].thrd = 0;
        workers[i].pending = false;
#endif
    }
    atomic_store_explicit(&n_barrier, 0, std::memory_order_relaxed);
    atomic_store_explicit(&n_barrier_passed, 0, std::memory_order_relaxed);
    this->parent_pool = parent_pool;
}

void nnml_threadgroup::destroy() {
    if (workers) {
        delete[] workers;
        workers = nullptr;
    }
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
}


// implementation of nnml_threadpool
void nnml_threadpool::init(int n_groups, int threads_per_group, nnml_affinity_mode affinity_mode) {
    total_groups = n_groups;
    view_groups  = n_groups;
    groups.reserve(n_groups);
    for (int i = 0; i < n_groups; ++i) {
        auto *grp = new nnml_threadgroup();
        grp->affinity_mode = affinity_mode;
        grp->init(threads_per_group, i, n_groups, this);
        groups.push_back(grp);
        total_threads += threads_per_group;
    }
}

void nnml_threadpool::destroy() {
    for (auto *grp : groups) {
        grp->destroy();
        delete grp;
    }
    groups.clear();
    total_groups = total_threads = 0;
}

void nnml_threadpool::set_view_groups(int n_view_groups) {
    // this function must be called after attach_work_buffer
    if (n_view_groups != 1 && n_view_groups != total_groups) {
        throw Error("Invalid view_groups count");
    }
    NNML_ASSERT(n_view_groups == 1 || n_view_groups == total_groups);
    NNML_ASSERT(main_buffer != nullptr);                                // must attach work buffer before set view groups

    view_groups = n_view_groups;
    // update the logical context of each group to know the current grouping
    for (int i = 0; i < total_groups; ++i) {
        groups[i]->crt_group_id = i % n_view_groups;
        for (int j = 0; j < groups[i]->n_threads_max; ++j)
            groups[i]->workers[j].nth = total_threads / n_view_groups;
        for (int j = 0; j < groups[i]->n_threads_max; ++j)
            groups[i]->workers[j].ith = groups[i]->workers[j].ith_global % groups[i]->workers[j].nth;
        for (int j = 0; j < groups[i]->n_threads_max; ++j) {
            if (n_view_groups == 1) groups[i]->workers[j].work_data = main_buffer;
            else groups[i]->workers[j].work_data = groups[i]->work_buffer;
        }
    }
}

nnml_threadgroup *nnml_threadpool::get_group(int id) {
    if (id < 0 || id >= total_groups) return nullptr;
    return groups[id];
}

void nnml_threadpool::attach_work_buffer(size_t work_size, uint8_t ** work_data) {          // work_size is temporarily same for all groups
    main_buffer = work_data[0];
    for (int i = 0; i < total_groups; ++i) {
        auto *grp = groups[i];
        grp->work_buffer = work_data[i];
        for (int t = 0; t < grp->n_threads_max; ++t) {
            grp->workers[t].work_size = work_size;
            grp->workers[t].work_data = work_data[i];
        }
    }
}

void nnml_threadpool::broadcast_stop() {                            // maybe thread-view unsafety
    for (int i = 0; i < total_groups; ++i) {
        auto *grp = groups[i];
        grp->stop.store(true, std::memory_order_release);
        pthread_cond_broadcast(&grp->cond);
    }
}

void nnml_threadpool::broadcast_resume() {                          // maybe thread-view unsafety
    for (int i = 0; i < total_groups; ++i) {
        auto *grp = groups[i];
        grp->pause.store(false, std::memory_order_release);
        pthread_cond_broadcast(&grp->cond);
    }
}


// Utility functions
void nnml_threadpool_kickoff(nnml_threadpool *pool, int n_threads_per_group) {
    if (!pool) return;

#ifdef NNML_USE_OPENMP
    assert(false);
#else
    // pthread path: set pending and broadcast
    for (int gid = 0; gid < pool->total_groups; ++gid) {
        auto * grp = pool->get_group(gid);
        if (!grp || !grp->work_fn) continue;

        int nth = (n_threads_per_group > 0 && n_threads_per_group <= grp->n_threads_max)
                  ? n_threads_per_group
                  : grp->n_threads_max;

        atomic_store_explicit(&grp->n_threads_cur, nth, std::memory_order_relaxed);

        // set the first n workers pending
        for (int j = 0; j < nth; ++j) {
            grp->workers[j].pending = true;
        }
        // wake up background threads
        pthread_cond_broadcast(&grp->cond);
    }
#endif
}

void nnml_threadgroup_run_worker0(nnml_threadgroup *grp) {
    // the argument grp must be valid and point to the first threadgroup
    if (!grp || !grp->work_fn) return;
#ifdef NNML_USE_OPENMP
    assert(false);
#else
    auto * st0 = &grp->workers[0];
    // if the worker0 is pending, here unset it
    if (st0->pending) st0->pending = false;
    grp->work_fn(st0);
    nnml_barrier(grp);
#endif
}

void nnml_threadpool_broadcast_stop(nnml_threadpool *pool) {
    if (pool) pool->broadcast_stop();
}

void nnml_threadpool_broadcast_resume(nnml_threadpool *pool) {
    if (pool) pool->broadcast_resume();
}

void nnml_barrier(struct nnml_threadgroup *tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, std::memory_order_relaxed);
    if (n_threads <= 1) return;

#ifdef NNML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, std::memory_order_relaxed);
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, std::memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        atomic_store_explicit(&tp->n_barrier, 0, std::memory_order_relaxed);
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, std::memory_order_seq_cst);
        return;
    }

    while (atomic_load_explicit(&tp->n_barrier_passed, std::memory_order_relaxed) == n_passed) {
        nnml_thread_cpu_relax();
    }

#ifdef NNML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, std::memory_order_seq_cst);
#else
    atomic_thread_fence(std::memory_order_seq_cst);
#endif
#endif
}

void nnml_barrier_global(struct nnml_threadpool *pool) {
    int total_threads = pool->total_threads;
    // printf("Total threads in global barrier: %d\n", total_threads);
    if (total_threads <= 1) return;

    int n_passed = atomic_load_explicit(&pool->n_barrier_passed_global, std::memory_order_relaxed);
    int n_barrier = atomic_fetch_add_explicit(&pool->n_barrier_global, 1, std::memory_order_seq_cst);

    if (n_barrier == total_threads - 1) {
        atomic_store_explicit(&pool->n_barrier_global, 0, std::memory_order_relaxed);
        atomic_fetch_add_explicit(&pool->n_barrier_passed_global, 1, std::memory_order_seq_cst);
        return;
    }

    while (atomic_load_explicit(&pool->n_barrier_passed_global, std::memory_order_relaxed) == n_passed)
        nnml_thread_cpu_relax();

#ifdef NNML_TSAN_ENABLED
    atomic_fetch_add_explicit(&pool->n_barrier_passed_global, 0, std::memory_order_seq_cst);
#else
    atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

int nnml_active_threads(const nnml_threadpool *pool) {
    // only count threads in the current view groups
    int active = 0;
    for (int i = 0; i < pool->total_groups; ++i) {
        const auto *grp = pool->groups[i];
        active += atomic_load_explicit(&grp->n_threads_cur, std::memory_order_relaxed);
    }
    return active;
}

static int nnml_set_affinity_platform(pthread_t th, int cpu_id) {
#if defined(__linux__)
    if (cpu_id < 0) return EINVAL;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int ret = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
    return ret;
#elif defined(_WIN32)
    HANDLE h = (HANDLE)th;
    if (!h || cpu_id < 0) return EINVAL;
    DWORD_PTR mask = 1ull << (cpu_id % 64);
    return SetThreadAffinityMask(h, mask) ? 0 : GetLastError();
#elif defined(__APPLE__)
    // macOS: no real CPU affinity
    thread_affinity_policy_data_t policy = { cpu_id };
    thread_port_t mach_thread = pthread_mach_thread_np(th);
    kern_return_t kr = thread_policy_set(mach_thread,
                                         THREAD_AFFINITY_POLICY,
                                         (thread_policy_t)&policy, 1);
    return (kr == KERN_SUCCESS) ? 0 : EINVAL;
#else
    NNML_UNUSED(th);
    NNML_UNUSED(cpu_id);
    return ENOTSUP;
#endif
}

int nnml_get_boot_node_id() {
    if (boot_node_id != -1) return boot_node_id;

    int local_var = 0;
    int local_node_idx = -1;
    if (get_mempolicy(&local_node_idx, NULL, 0, &local_var, MPOL_F_ADDR | MPOL_F_NODE) == 0) {
        boot_node_id = local_node_idx;
        return local_node_idx;
    }
    fprintf(stderr, "Warning: cannot determine local NUMA node.\n");
    return -1;
}


// APIs
#ifndef NNML_USE_OPENMP
static void * nnml_worker_entry(void * arg) {
    auto * st  = static_cast<nnml_compute_state *>(arg);
    auto * grp = st->threadgrp;

    while (true) {
        // pause loop
        while (grp->pause.load(std::memory_order_acquire)) {
            pthread_mutex_lock(&grp->mutex);
            if (grp->pause.load(std::memory_order_acquire) && !grp->stop.load(std::memory_order_acquire)) {
                pthread_cond_wait(&grp->cond, &grp->mutex);
            }
            pthread_mutex_unlock(&grp->mutex);
            if (grp->stop.load(std::memory_order_acquire)) break;
        }
        if (grp->stop.load(std::memory_order_acquire)) break;
        // wait to be kicked
        bool do_work = false;
        if (st->pending) {
            do_work = true;
            st->pending = false;
        } else {
            // if no pending work, briefly yield
            nnml_thread_cpu_relax();
        }
        if (do_work) {
            if (grp->work_fn) grp->work_fn(st);
            // all workers must reach the barrier
            nnml_barrier(grp);
        }
    }
    return nullptr;
}
#endif // !NNML_USE_OPENMP

NNML_API nnml_threadpool * nnml_threadpool_new(int n_groups,
                                    int threads_per_group,
                                    bool start_paused,
                                    nnml_affinity_mode affinity_mode,
                                    nnml_compute_fn fn) {
    // create the thread pool object and initialize it
    auto *pool = new nnml_threadpool();
    pool->init(n_groups, threads_per_group, affinity_mode);

    // initialize each group and create worker threads
    for (int gid = 0; gid < n_groups; ++gid) {
        auto *grp = pool->groups[gid];
        grp->work_fn = fn;
        grp->pause.store(start_paused, std::memory_order_relaxed);
        grp->stop.store(false, std::memory_order_relaxed);
        grp->abort.store(0, std::memory_order_relaxed);
        atomic_store_explicit(&grp->n_threads_cur, threads_per_group, std::memory_order_relaxed);

        // create worker threads for this group
        //        group 0 -> worker0 = main thread
        //        group 0 -> worker1..N-1 = background threads
        //        group 1..G-1 -> worker0..N-1 = all background threads
        int start_index = (gid == 0) ? 1 : 0;  // group0 reserves worker0 for the main thread
        for (int j = start_index; j < grp->n_threads_max; ++j) {
            grp->workers[j].threadgrp = grp;
            int rc = pthread_create(&grp->workers[j].thrd, nullptr,
                                    nnml_worker_entry,
                                    &grp->workers[j]);
            if (rc != 0) {
                fprintf(stderr, "Failed to create worker %d in group %d\n", j, gid);
                exit(EXIT_FAILURE);
            }
        }
    }
    pool->groups[0]->workers[0].threadgrp = pool->groups[0]; // main thread worker0 points to the correct threadgroup
    pool->groups[0]->workers[0].thrd = pthread_self();
    
    if(nnml_threadpool_bind_affinity(pool, affinity_mode) != 0) {
        fprintf(stderr, "Warning: Failed to bind threadpool affinity (mode=%d)\n", (int)affinity_mode);
    }
    return pool;
}

NNML_API void nnml_threadpool_destroy(nnml_threadpool *pool) {
    if (pool) pool->destroy();
}

NNML_API void nnml_threadpool_pause(nnml_threadpool *pool) {
    if (!pool) return;
    for (int gid = 0; gid < (int)pool->groups.size(); ++gid) {
        auto * grp = pool->groups[gid];
        if (!grp) continue;
        grp->pause.store(true, std::memory_order_release);
    }
}

NNML_API void nnml_threadpool_resume(nnml_threadpool *pool) {
    if (!pool) return;
    for (int gid = 0; gid < (int)pool->groups.size(); ++gid) {
        auto * grp = pool->groups[gid];
        if (!grp) continue;
        grp->pause.store(false, std::memory_order_release);
#ifndef NNML_USE_OPENMP
        pthread_cond_broadcast(&grp->cond);
#endif
    }
}

NNML_API void nnml_threadpool_stop(nnml_threadpool *pool) {
    if (!pool) return;
    for (int gid = 0; gid < (int)pool->groups.size(); ++gid) {
        auto * grp = pool->groups[gid];
        if (!grp) continue;
        grp->stop.store(true, std::memory_order_release);
#ifndef NNML_USE_OPENMP
        pthread_cond_broadcast(&grp->cond);
        // stop the workers, worker0 is main thread, so only join worker1..N-1
        int start_index = (gid == 0) ? 1 : 0;
        for (int j = start_index; j < grp->n_threads_max; ++j) {
            if (grp->workers[j].thrd) {
                pthread_join(grp->workers[j].thrd, nullptr);
                grp->workers[j].thrd = 0;
            }
        }
#endif
    }
}

NNML_API nnml_topology_info detect_topology() {
    nnml_topology_info topo;
    // CPU count
#if defined(__linux__) || defined(__APPLE__)
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    topo.cpu_count = nproc > 0 ? (int)nproc : 1;
#elif defined(_WIN32)
    DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    topo.cpu_count = count > 0 ? (int)count : 1;
#else
    topo.cpu_count = 1;
#endif

    topo.numa_nodes = 1;
    topo.nodes.clear();
#if defined(__linux__) && NNML_HAS_NUMA
    if (numa_available() != -1) {
        int max_node = numa_max_node(); // highest index
        topo.numa_nodes = max_node + 1;

        for (int nid = 0; nid < topo.numa_nodes; ++nid) {
            nnml_numanode_info info;
            info.node_id = nid;

            struct bitmask *cpumask = numa_allocate_cpumask();
            if (numa_node_to_cpus(nid, cpumask) == 0) {
                for (int i = 0; i < cpumask->size; ++i) {
                    if (numa_bitmask_isbitset(cpumask, i)) {
                        info.cpus.push_back(i);
                    }
                }
            }
            numa_free_cpumask(cpumask);
            topo.nodes.push_back(std::move(info));
        }

        int total = 0;
        for (auto &n : topo.nodes) total += (int)n.cpus.size();
        if (total == 0) {
            // fallback even split
            int per = std::max(1, topo.cpu_count / topo.numa_nodes);
            int cur = 0;
            topo.nodes.clear();
            for (int nid = 0; nid < topo.numa_nodes; ++nid) {
                nnml_numanode_info info;
                info.node_id = nid;
                int take = (nid == topo.numa_nodes - 1) ? (topo.cpu_count - cur) : per;
                for (int k = 0; k < take; ++k) info.cpus.push_back(cur++);
                topo.nodes.push_back(std::move(info));
            }
        }
    } else
#endif
#if defined(_WIN32)
    {
        // Windows NUMA
        ULONG highest = 0;
        if (GetNumaHighestNodeNumber(&highest)) {
            topo.numa_nodes = (int)highest + 1;
            topo.nodes.resize(topo.numa_nodes);
            for (int nid = 0; nid < topo.numa_nodes; ++nid) {
                topo.nodes[nid].node_id = nid;
#if defined(GetNumaNodeProcessorMaskEx)
                GROUP_AFFINITY ga = {};
                if (GetNumaNodeProcessorMaskEx((USHORT)nid, &ga)) {
                    // only use the 64-bit mask for the first group
                    KAFFINITY mask = ga.Mask;
                    for (int b = 0; b < 64; ++b) {
                        if (mask & (1ull << b)) topo.nodes[nid].cpus.push_back(b);
                    }
                }
#endif
            }
            // if no cpus detected, fallback to even split
            int total = 0;
            for (auto &n : topo.nodes) total += (int)n.cpus.size();
            if (total == 0) {
                int cur = 0;
                int per = std::max(1, topo.cpu_count / topo.numa_nodes);
                for (int nid = 0; nid < topo.numa_nodes; ++nid) {
                    int take = (nid == topo.numa_nodes - 1) ? (topo.cpu_count - cur) : per;
                    for (int k = 0; k < take; ++k) topo.nodes[nid].cpus.push_back(cur++);
                }
            }
        } else {
            topo.numa_nodes = 1;
        }
    }
#else
    {
        // macOS or others: single node
        topo.numa_nodes = 1;
        nnml_numanode_info info;
        info.node_id = 0;
        for (int i = 0; i < topo.cpu_count; ++i) info.cpus.push_back(i);
        topo.nodes.push_back(std::move(info));
    }
#endif
    // if no cpus detected, fallback to single node
    if (topo.nodes.empty()) {
        nnml_numanode_info info;
        info.node_id = 0;
        for (int i = 0; i < topo.cpu_count; ++i) info.cpus.push_back(i);
        topo.nodes.push_back(std::move(info));
        topo.numa_nodes = 1;
    }

    return topo;
}

NNML_API int nnml_threadpool_bind_affinity(nnml_threadpool *pool, nnml_affinity_mode mode) {
    if (!pool) return -1;
    if (mode == nnml_affinity_mode::NO_BIND) return 0;

    nnml_topology_info topo = detect_topology();
    std::vector<int> all_cpus;
    for (auto &node : topo.nodes)
        for (int c : node.cpus)
            all_cpus.push_back(c);

    if (all_cpus.empty()) {
        for (int i = 0; i < topo.cpu_count; ++i) all_cpus.push_back(i);
    }

    std::vector<pthread_t> threads;
    std::map<pthread_t, nnml_compute_state*> thread_state_map;
    threads.reserve(pool->total_threads);
    for (auto *grp : pool->groups)
        for (int i = 0; i < grp->n_threads_max; ++i) {
            threads.push_back(grp->workers[i].thrd);
            thread_state_map[grp->workers[i].thrd] = &(grp->workers[i]);
        }

    if (mode == nnml_affinity_mode::NUMA_UNIFIED) {
        if ((int)threads.size() > (int)all_cpus.size()) {
            NNML_ASSERT_MSG(false, "Not enough CPUs for NUMA_UNIFIED binding, more threads (than CPUs) will slow down inference!");
            return -1;
        }

        int local_node_idx = nnml_get_boot_node_id();

        std::vector<int> ordered_cpus;
        ordered_cpus.reserve(all_cpus.size());
        if (local_node_idx != -1) {
            // first add the local node's CPUs
            for (int c : topo.nodes[local_node_idx].cpus) {
                ordered_cpus.push_back(c);
            }
            // then add other nodes' CPUs
            for (int i = 0; i < (int)topo.nodes.size(); ++i) {
                if (i == local_node_idx) continue;
                for (int c : topo.nodes[i].cpus) {
                    ordered_cpus.push_back(c);
                }
            }
        } else {
            // if cannot determine local node, just use the original order
            ordered_cpus = all_cpus;
        }

        fprintf(stdout, "Binding threads in NUMA_UNIFIED mode, local_node_idx=%d\n", local_node_idx);
        for (size_t i = 0; i < threads.size(); ++i) {
            int rc = nnml_set_affinity_platform(threads[i], ordered_cpus[i]);
            if (rc == 0) thread_state_map[threads[i]]->cpu_core = ordered_cpus[i];
            else return -1;
        }
        return 0;
    }

    if (mode == nnml_affinity_mode::NUMA_DISTRIBUTED) {
        NNML_ASSERT_MSG(threads.size() % topo.numa_nodes == 0, "total_threads must be times of nodes");
        int nodes = topo.numa_nodes;
        NNML_ASSERT(nodes >= pool->total_groups);
        nodes = std::min(nodes, pool->total_groups);

        int total_threads = threads.size();
        int base = total_threads / nodes;
        int idx = 0;
        for (int nid = 0; nid < nodes; ++nid) {
            const auto &node = topo.nodes[nid];
            if (base > (int)node.cpus.size()) NNML_ABORT("Not enough CPUs in NUMA node for NUMA_DISTRIBUTED binding");

            for (int j = 0; j < base; ++j, ++idx) {
                int rc = nnml_set_affinity_platform(threads[idx], node.cpus[j]);
                if (rc == 0) thread_state_map[threads[idx]]->cpu_core = node.cpus[j];
                if (rc != 0) return -1;
            }
        }
        return 0;
    }

    return -1;
}

NNML_API void nnml_print_cpu_bindings(nnml_threadpool *pool) {
    if (!pool) return;
    printf("--------------- Threadpool CPU Bindings ---------------\n");
    for (auto *grp : pool->groups) {
        for (int i = 0; i < grp->n_threads_max; ++i) {
            auto &st = grp->workers[i];
            printf("Group %d, Worker %d (Thread ID: %p): bound to CPU %d\n",
                   grp->crt_group_id, st.ith, (void*)st.thrd, st.cpu_core);
        }
    }
    printf("-------------------------------------------------------\n");
}

NNML_API const char * nnml_status_to_string(enum nnml_status status) {
    switch (status) {
        case NNML_STATUS_ALLOC_FAILED: return "NNML status: error (failed to allocate memory)";
        case NNML_STATUS_FAILED:       return "NNML status: error (operation failed)";
        case NNML_STATUS_SUCCESS:      return "NNML status: success";
        case NNML_STATUS_ABORTED:      return "NNML status: warning (operation aborted)";
    }
    return "NNML status: unknown";
}
