/**
 * @file thread.h
 * @brief thread pool and affinity management for NNML.
 * 
 * Provides:
 *  - A thread pool abstraction supporting both standard threads and OpenMP (please wait).
 *  - The thread pool support multi-view grouping, where each view can have its own process.
 *  - NUMA-aware thread affinity management with unified and distributed modes.
 *  - Efficient barrier synchronization for worker threads.
 * 
 * Designed for inference workloads requiring high parallelism and NUMA locality control.
 */
#pragma once
#include <pthread.h>
#include <atomic>
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>

#include "nnml.h"


#define NNML_CACHE_LINE 64
#define NNML_MAX_THREADS 512
#define NNML_MAX_THREAD_GROUPS 8

#if defined(_MSC_VER)
#define NNML_CACHE_ALIGN __declspec(align(NNML_CACHE_LINE))
#elif defined(__clang__) || defined(__GNUC__)
#define NNML_CACHE_ALIGN __attribute__((aligned(NNML_CACHE_LINE)))
#else
#define NNML_CACHE_ALIGN
#endif

#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define NNML_TSAN_ENABLED 1
#  endif
#elif defined(__SANITIZE_THREAD__)
#  define NNML_TSAN_ENABLED 1
#endif

#if __has_include(<numa.h>)
  #include <numa.h>
  #include <numaif.h>
  #define NNML_HAS_NUMA 1
#else
  #define NNML_HAS_NUMA 0
#endif

#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
inline void nnml_thread_cpu_relax() { __asm__ volatile("yield" ::: "memory"); }
#elif defined(__x86_64__)
#include <immintrin.h>
inline void nnml_thread_cpu_relax() { _mm_pause(); }
#else
inline void nnml_thread_cpu_relax() {}
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(NNML_CACHE_LINE))
typedef volatile LONG atomic_int;
typedef atomic_int    atomic_bool;
typedef atomic_int    atomic_flag;
#define ATOMIC_FLAG_INIT 0
typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;
void atomic_store(atomic_int *ptr, LONG val);
void atomic_store_explicit(atomic_int *ptr, LONG val, memory_order mo);
LONG atomic_load(atomic_int *ptr);
LONG atomic_load_explicit(atomic_int *ptr, memory_order mo);
LONG atomic_fetch_add(atomic_int *ptr, LONG inc);
LONG atomic_fetch_add_explicit(atomic_int *ptr, LONG inc, memory_order mo);
atomic_bool atomic_flag_test_and_set(atomic_flag *ptr);
void atomic_flag_clear(atomic_flag *ptr);
void atomic_thread_fence(memory_order mo);
#else
#include <atomic>
#endif
#else
#include <atomic>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif


typedef pthread_cond_t  nnml_cond_t;
typedef pthread_mutex_t nnml_mutex_t;
typedef pthread_t       nnml_thread_t;

enum nnml_status {                      // thread pool status codes
    NNML_STATUS_ALLOC_FAILED = -2,
    NNML_STATUS_FAILED       = -1,
    NNML_STATUS_SUCCESS      = 0,
    NNML_STATUS_ABORTED      = 1,
};

enum nnml_affinity_mode {               // thread affinity modes
    NO_BIND          = -1,              // no affinity control, let OS decide
    NUMA_UNIFIED     = 0,               // NNML_UNIFIED and NNML_DISTRIBUTED are used in nnml_threadpool_bind_affinity()
    NUMA_DISTRIBUTED = 1
};

enum nnml_compute_backend {
    STD_THREADS = 0,
    OPENMP      = 1                     // reserved for future and only support NUMA_UNIFIED, not implemented yet
};

struct nnml_cpu_set {
#if defined(__linux__)
    cpu_set_t set{};
#endif
    std::vector<int> cpus;
};

struct nnml_numanode_info {
    int node_id = -1;
    std::vector<int> cpus;
};

struct nnml_topology_info {
    int cpu_count  = 0;
    int numa_nodes = 1;
    std::vector<nnml_numanode_info> nodes;
};

struct nnml_threadgroup;            // pre-declaration
struct nnml_threadpool;
using  nnml_compute_fn = void (*)(void *);


/** nnml_compute_state: state for each worker thread, passed to the worker function.
 *   - Contains thread id, group size, global id and global size, CPU affinity info, and a pointer to the thread group.
 *   - The worker function can use this state to determine its role and access shared resources.
 */
struct nnml_compute_state {
#ifndef NNML_USE_OPENMP
    nnml_thread_t thrd;
    bool pending = false;
#endif
    int  cpu_core = -1;
    bool cpumask[NNML_MAX_THREADS] = {false};
    nnml_threadgroup * threadgrp = nullptr;
    int  nth = 0;
    int  ith = 0;
    int  ith_global = 0;
    int  nth_global = 0;

    size_t    work_size;            // size of tmp work buffer
    uint8_t * work_data;
};


/** nnml_threadgroup: represents a group of worker threads that can synchronize together.
 *   - Contains synchronization primitives (mutex, condition variable, barriers).
 *   - Contains an array of compute states for its worker threads.
 *   - Contains the compute function pointer that all workers in the group will execute.
 *   - Contains affinity mode and a pointer to the parent thread pool for shared resources.
 */
struct nnml_threadgroup {
    nnml_mutex_t mutex;
    nnml_cond_t  cond;

    std::atomic<int> NNML_CACHE_ALIGN n_barrier{0};
    std::atomic<int> NNML_CACHE_ALIGN n_barrier_passed{0};
    std::atomic<int> NNML_CACHE_ALIGN current_chunk;

    std::atomic<bool> stop{false};
    std::atomic<bool> pause{false};
    std::atomic<int>  abort{0};

    nnml_compute_state *workers = nullptr;

    std::atomic<int> n_threads_cur{0};
    int       n_threads_max = 0;
    int       n_groups      = 0;       // the only point to know how many groups there are
    int       crt_group_id  = 0;
    int32_t   prio          = 0;
    uint32_t  poll          = 0;
    uint8_t * work_buffer   = nullptr;

    nnml_compute_fn    work_fn       = nullptr;
    nnml_affinity_mode affinity_mode = nnml_affinity_mode::NO_BIND;
    nnml_threadpool *  parent_pool   = nullptr;
    nnml_status        ec            = NNML_STATUS_SUCCESS;

    nnml_threadgroup() = default;
    void init(int n_threads, int group_id = 0, int total_groups = 1, nnml_threadpool * parent_pool = nullptr);
    void destroy();
};


/** nnml_threadpool: represents the entire thread pool, containing multiple thread groups.
 *   - Contains a vector of pointers to thread groups.
 *   - Contains total thread and group counts, and a main buffer for shared work data.
 *   - Contains global_barrier counters for synchronizing across all threads in the pool.
 *   - Provides APIs to initialize, destroy, set view groups, broadcast stop/resume signals, and attach work buffers.
 */
struct nnml_threadpool {
    std::vector<nnml_threadgroup *> groups;
    uint8_t * main_buffer = nullptr;        // always the 0-th work buffer in the pool

    int total_threads = 0;
    int total_groups  = 0;
    int view_groups   = 0;

    std::atomic<int> NNML_CACHE_ALIGN n_barrier_global{0};
    std::atomic<int> NNML_CACHE_ALIGN n_barrier_passed_global{0};

    nnml_threadpool() = default;
    void init(int n_groups, int threads_per_group, nnml_affinity_mode affinity_mode);
    void destroy();

    void set_view_groups(int n_view_groups);
    int  get_total_view_groups() const { return view_groups; }
    nnml_threadgroup *get_group(int id);

    void broadcast_stop();
    void broadcast_resume();

    void attach_work_buffer(size_t work_size, uint8_t ** work_data);
};


// Utility functions
void       nnml_threadpool_kickoff(nnml_threadpool *pool, int n_threads_per_group);
void       nnml_threadgroup_run_worker0(nnml_threadgroup *grp);
void       nnml_threadpool_broadcast_stop(nnml_threadpool *pool);
void       nnml_threadpool_broadcast_resume(nnml_threadpool *pool);

void       nnml_barrier(struct nnml_threadgroup *tp);
void       nnml_barrier_global(struct nnml_threadpool *pool);

int        nnml_active_threads(const nnml_threadpool *pool);
static int nnml_set_affinity_platform(pthread_t th, int cpu_id);
// static int nnml_get_affinity_platform(pthread_t th);

extern int boot_node_id;
int        nnml_get_boot_node_id();

// APIs
NNML_API nnml_threadpool * nnml_threadpool_new(int n_groups, int threads_per_group, bool start_paused, nnml_affinity_mode mode, nnml_compute_fn fn);
NNML_API void nnml_threadpool_destroy(nnml_threadpool *pool);
NNML_API void nnml_threadpool_pause(nnml_threadpool *pool);
NNML_API void nnml_threadpool_resume(nnml_threadpool *pool);
NNML_API void nnml_threadpool_stop(nnml_threadpool *pool);

NNML_API nnml_topology_info detect_topology();
NNML_API int  nnml_threadpool_bind_affinity(nnml_threadpool *pool, nnml_affinity_mode mode);
NNML_API void nnml_print_cpu_bindings(nnml_threadpool *pool);

NNML_API const char * nnml_status_to_string(enum nnml_status status);
