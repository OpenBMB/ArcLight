/**
 * @file memory.h
 * @brief memory management unit for NNML.
 *
 * Provides:
 *  - Fixed-size aligned buffer
 *  - Elastic (growable) buffer
 *  - Buffer view abstraction
 *  - Dual-buffer utility
 *  - NUMA-aware memory manager
 *
 * Designed for inference workloads requiring deterministic allocation and NUMA locality control.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <type_traits>
#include <unistd.h>

#include "nnml.h"
#include "tensor.h"


/**
 * Alignment policy:
 *  - 4 bytes for 32-bit systems
 *  - 16 bytes for 64-bit systems
 */
#if UINTPTR_MAX == 0xFFFFFFFF
    #define NNML_MEM_ALIGN 4
#else
    #define NNML_MEM_ALIGN 16
#endif

#define NNML_ALIGN_UP(x)  (((x) + (NNML_MEM_ALIGN - 1)) & ~(NNML_MEM_ALIGN - 1))

#define NNML_ELASTIC_BUFFER_MIN_ALLOC_SIZE  (1024 * 1024 * 1024)       // now it is not used
#define NNML_ELASTIC_BUFFER_GROWTH_SIZE  (10 * 1024 * 1024)
#define NNML_MAX_GROWTH_TRIES 3


enum nnml_buffer_usage {
    NNML_BUFFER_USAGE_ANY     = 0,      // no specific usage
    NNML_BUFFER_USAGE_WEIGHTS = 1,      // weight tensors, typically protected after initialization
    NNML_BUFFER_USAGE_COMPUTE = 2,      // activation tensors, include kvcache
};

enum nnml_alloc_strategy {
    NNML_ALLOC_STRATEGY_UMA  = 0,       // unified memory allocation, also suitable for NUMA platforms
    NNML_ALLOC_STRATEGY_NUMA = 1,       // partitioned NUMA-aware allocation, better for NUMA platforms
};

struct nnml_numa_info {
    bool     available  = false;
    uint32_t node_count = 1;
};


/**
 * nnml_buffer: fixed-size aligned buffer with simple bump-pointer allocation.
 *  - Supports NUMA-aware allocation on Linux with libnuma.
 *  - Provides basic read/write/memset operations with bounds checking.
 *  - Does not support deallocation of individual objects; use clear() to reset.
 *  - Designed for deterministic memory usage patterns in inference workloads.
 */
class nnml_buffer {
public:
    nnml_buffer() = default;
    ~nnml_buffer() { release(); }

    nnml_buffer(std::size_t size, bool zero_init, int numa_node, nnml_buffer_usage usage, std::size_t alignment);
    nnml_buffer(const nnml_buffer&)                      = delete;
    nnml_buffer(nnml_buffer&& other)            noexcept = delete;

    nnml_buffer& operator=(const nnml_buffer&)           = delete;
    nnml_buffer& operator=(nnml_buffer&& other) noexcept = delete;

    virtual void  allocate(std::size_t size_bytes, bool zero_init = false);
    virtual void* allocate_obj(size_t bytes);
    virtual void  clear();
    virtual void  release();

    void*               base()              { return base_; }
    const void*         base()        const { return base_; }
    virtual std::size_t size()        const { return size_; }
    virtual std::size_t used_size()   const { return offset_; }
    std::size_t         offset()      const { return offset_; }
    std::size_t         remaining()   const { return size_ - offset_; }
    int                 numa_node()   const { return numa_node_; }
    nnml_buffer_usage   usage()       const { return usage_; }

protected:
    int               numa_node_   = -1;
    nnml_buffer_usage usage_       = NNML_BUFFER_USAGE_ANY;
    std::size_t       alignment_   = NNML_MEM_ALIGN;

private:
    void*       base_   = nullptr;
    std::size_t size_   = 0;
    std::size_t offset_ = 0;
};

typedef nnml_buffer * nnml_buffer_t;


/** nnml_elastic_buffer: a growable buffer that manages multiple contiguous regions.
 *   - Each region is allocated as a separate block.
 *   - Provides allocate_obj API that tries to allocate from the last region and grows if needed.
 *   - clear() releases all regions except the first one and resets it for reuse.
 *   - Designed for use cases like KV cache where the total size may grow beyond initial estimates.
 */
class nnml_elastic_buffer : public nnml_buffer {
public:
    nnml_elastic_buffer() = default;
    ~nnml_elastic_buffer() { release(); }

    nnml_elastic_buffer(std::size_t size, bool zero_init, int numa_node, nnml_buffer_usage usage, std::size_t alignment);
    nnml_elastic_buffer(const nnml_elastic_buffer&)                      = delete;
    nnml_elastic_buffer(nnml_elastic_buffer&& other)            noexcept = delete;

    nnml_elastic_buffer& operator=(const nnml_elastic_buffer&)           = delete;
    nnml_elastic_buffer& operator=(nnml_elastic_buffer&& other) noexcept = delete;

    void  allocate(std::size_t size_bytes, bool zero_init = false) override;
    void* allocate_obj(size_t bytes) override;
    void  release() override;
    void  clear();

    void*       base(unsigned idx = 0)            { return region_base(idx); }
    const void* base(unsigned idx = 0)      const { return region_base(idx); }
    std::size_t size(unsigned idx)          const { return region_size(idx); }
    std::size_t offset(unsigned idx = 0)    const { return region_offset(idx); }
    std::size_t remaining(unsigned idx = 0) const { return region_size(idx) - region_offset(idx); }
    std::size_t size()                      const override;
    std::size_t used_size()                 const override;
    uint32_t    n_regions()                 const { return static_cast<uint32_t>(regions_.size()); }

private:
    struct buffer_region {
        void*       base_   = nullptr;
        std::size_t size_   = 0;
        std::size_t offset_ = 0;
    };

    std::vector<buffer_region> regions_;

    void*       region_base  (unsigned idx) const;
    std::size_t region_size  (unsigned idx) const;
    std::size_t region_offset(unsigned idx) const;
};


/** nnml_buffer_view: a non-owning view into a contiguous memory region (e.g., from nnml_buffer or nnml_elastic_buffer).
 *   - Provides allocate_obj API that allocates from the view with bounds checking.
 *   - Does not support deallocation; use clear() to reset the view.
 *   - Designed for sub-allocations within a larger buffer, as it maintains its own base and offset.
 */
class nnml_buffer_view {
public:
    nnml_buffer_view() = default;
    nnml_buffer_view(void* base, std::size_t size, std::size_t alignment = NNML_MEM_ALIGN)
        : base_(base), size_(size), offset_(0), alignment_(alignment) {}

    void*       base()      const { return base_; }
    std::size_t size()      const { return size_; }
    std::size_t offset()    const { return offset_; }
    std::size_t remaining() const { return size_ - offset_; }

    void* allocate_obj(std::size_t bytes);
    void clear();           // does not modify the underlying memory, just resets the offset for reuse.

private:
    void*       base_      = nullptr;
    std::size_t size_      = 0;
    std::size_t offset_    = 0;
    std::size_t alignment_ = NNML_MEM_ALIGN;
};


/** nnml_dual_buffer: utility that reserves two contiguous object blocks from a parent buffer.
 *   - The parent buffer can be either nnml_buffer or nnml_elastic_buffer (only the first region is used for allocation).
 *   - Exposes two nnml_buffer_view instances representing the two sub-buffers.
 *   - Provides allocate_obj_on(idx) and clear_on(idx) to operate on the selected sub-buffer.
 *   - Designed for use cases like double-buffering activations where two separate buffers are needed.
 *   - Usage: call allocate_obj_on(0) to allocate from the first sub-buffer, and allocate_obj_on(1) for the second.
 */
class nnml_dual_buffer {
public:
    nnml_dual_buffer(nnml_buffer& parent, std::size_t size, std::size_t alignment = NNML_MEM_ALIGN);
    nnml_buffer_view&       view(unsigned idx);
    const nnml_buffer_view& view(unsigned idx) const;

    void* allocate_obj_on(unsigned idx, std::size_t bytes);      // allocate on selected sub-buffer (0 or 1)
    void  clear_on(unsigned idx);                                // clear selected sub-buffer

private:
    nnml_buffer_view views_[2];
    std::size_t      alignment_ = NNML_MEM_ALIGN;
};


/** node_buffers: a simple struct to hold the set of buffers for a single NUMA node.
 *   - Contains separate buffers for weights, activations, kv cache, and temporary workspace.
 *   - Designed for easy access to the different buffer types associated with each NUMA node.
 */
struct node_buffers {
    nnml_buffer * w_buffer;
    nnml_buffer * a_buffer;
    nnml_buffer * kv_buffer;
    nnml_buffer * tmp_work_buffer;

    node_buffers() = default;
    node_buffers(nnml_buffer * w, nnml_buffer * a, nnml_buffer * kv, nnml_buffer * tmp_work)
        : w_buffer(w), a_buffer(a), kv_buffer(kv), tmp_work_buffer(tmp_work) {}
};


/** nnml_memory: high-level memory manager that encapsulates multiple node_buffers and dual_buffers.
 *  - Manages the overall memory allocation strategy (UMA vs NUMA) and buffer organization.
 *  - Provides allocate_obj API that routes to the appropriate buffer based on tensor type and buffer ID.
 *  - Provides utility functions for querying total and used sizes across all buffers.
 *  - Designed as the main interface for memory management in NNML, abstracting away the details of individual buffers.
 */
class nnml_memory {
public:
    struct cparams {
        nnml_alloc_strategy strategy            = NNML_ALLOC_STRATEGY_UMA;
        int                 n_nodes_manual      = -1;
        std::size_t         total_wsize_bytes   = 0;               // total weight buffer size across all nodes
        std::size_t         total_csize_bytes   = 0;
        std::size_t         total_kvsize_bytes  = 0;
        std::size_t         tmp_work_size_bytes = 0;
        std::size_t         dual_buffer_single_size = 0;           // not the total size, but the size of each
        std::size_t         alignment           = NNML_MEM_ALIGN;
        bool                zero_init           = false;
        bool                is_kv_elastic       = false;
        bool                build_dual_buffer   = false;
        // 
    };

    nnml_memory() = default;
    ~nnml_memory() { release(); };

    nnml_memory(const nnml_memory&)            = delete;
    nnml_memory(nnml_memory&&)                 = delete;

    nnml_memory& operator=(const nnml_memory&) = delete;
    nnml_memory& operator=(nnml_memory&&)      = delete;

    // creation and release
    static nnml_memory_t create(const cparams& params);
    void                 release();
    int                  create_dual_buffers(std::size_t single_size);
    int                  clear_activation_buffers();
    int                  clear_kv_buffers();
    // access interfaces
    const std::vector<node_buffers>& buffers() const { return buffers_; }
    nnml_buffer_t                    weight_buffer(unsigned node = 0);
    nnml_buffer_t                    activation_buffer(unsigned node = 0);
    nnml_buffer_t                    kv_buffer(unsigned node = 0);
    nnml_buffer_t                    tmp_work_buffer(unsigned node = 0);
    nnml_dual_buffer&                dual_buffer(unsigned node = 0);
    // for UMA allocation
    nnml_buffer_t                    single_weight_buffer();
    nnml_buffer_t                    single_activation_buffer();
    nnml_buffer_t                    single_kv_buffer();
    nnml_buffer_t                    single_tmp_work_buffer();

    std::size_t total_size()                 const;
    std::size_t total_weight_size()          const;
    std::size_t total_activation_size()      const;
    std::size_t total_kv_size()              const;
    std::size_t total_used_size()            const;
    std::size_t total_used_weight_size()     const;
    std::size_t total_used_activation_size() const;
    std::size_t total_used_kv_size()         const;

    nnml_numa_info numa_info() const { return numa_info_; }

    // allocate an object from the appropriate buffer
    void* allocate_obj(nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, size_t bytes);

private:
    std::size_t                   alignment_ = NNML_MEM_ALIGN;
    std::vector<node_buffers>     buffers_;
    std::vector<nnml_dual_buffer> dual_buffers_;
    nnml_numa_info                numa_info_{};
};

#ifdef  __cplusplus
extern "C" {
#endif

NNML_API nnml_numa_info query_numa_info();

#ifdef  __cplusplus
}
#endif
