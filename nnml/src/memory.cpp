/** 
 * @file memory.cpp
 * @brief Implementation of memory.h
 * 
 * This file implements the nnml_memory class which manages memory buffers for weights, activations, and kvcaches.
 * It supports both UMA and NUMA allocation strategies, as well as elastic buffers that can grow dynamically.
 * The implementation uses platform-specific APIs for memory allocation and NUMA support.
 */
#include <new>
#include <algorithm>
#include <mutex>
#include <assert.h>

#if defined(__linux__)
  #define NNML_HAS_LIBNUMA 1
  #include <unistd.h>
  #include <sys/mman.h>
  #include <errno.h>
  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
  #include <atomic>
  #include <malloc.h>
  #ifdef NNML_HAS_LIBNUMA
    #include <numa.h>
  #endif
#elif defined(_WIN32)
  #include <windows.h>
  #include <malloc.h>
#else
  #include <cstdlib>
#endif

#include "nnml.h"
#include "memory.h"


// NNML_API for query_numa_info()
NNML_API nnml_numa_info query_numa_info() {
    nnml_numa_info info{};
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    if (numa_available() != -1) {
        info.available = true;
        info.node_count = static_cast<uint32_t>(numa_max_node() + 1);
    } else {
        info.available = false;
        info.node_count = 1;
    }
#else
    NNML_UNUSED(numa_mode);
    info.available = false;
    info.node_count = 1;
#endif
    return info;
}


// inline functions for platform-specific aligned allocation and NUMA support
inline std::size_t nnml_get_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
#else
    return static_cast<std::size_t>(::getpagesize());
#endif
}

inline void* aligned_alloc_bytes(std::size_t size, std::size_t alignment_) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment_);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
    void* p = nullptr;
    if (alignment_ < sizeof(void*)) alignment_ = sizeof(void*);
    if (posix_memalign(&p, alignment_, size) != 0) return nullptr;
    return p;
#else
    return std::aligned_alloc(alignment_, ((size + alignment_ - 1) / alignment_) * alignment_);
#endif
}

inline void aligned_free_bytes(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

inline bool numa_is_available() {
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    static int once = numa_available();      // -1 means not available
    return once != -1;
#else
    return false;
#endif
}

inline void* numa_alloc_on_node(std::size_t size, int node) {
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    if (!numa_is_available()) return nullptr;
    return ::numa_alloc_onnode(size, node);
#else
    NNML_USED(size);
    NNML_USED(node);
    return nullptr;
#endif
}

inline void numa_free_on_node(void* p, std::size_t size) {
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    ::numa_free(p, size);
#else
    NNML_USED(p);
    NNML_USED(size);
#endif
}


// implementation of nnml_buffer
nnml_buffer::nnml_buffer(std::size_t size, bool zero_init, int numa_node, nnml_buffer_usage usage, std::size_t alignment)
    : size_(size), numa_node_(numa_node), usage_(usage), alignment_(alignment) {
    if ((alignment & (alignment - 1)) != 0) {
        throw Error("nnml_buffer::allocate(): alignment must be power of two");
    }
    alignment_ = std::max(alignment, nnml_get_page_size());
    // Initialize memory allocation logic
    allocate(size, zero_init);
}

void nnml_buffer::allocate(std::size_t size_bytes, bool zero_init) {
    if (base_) {
        throw Error("nnml_buffer::allocate(): already allocated or adopted");
    }
    void* p = nullptr;
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    if ((numa_node_ >= 0) && numa_is_available()) {
        p = numa_alloc_on_node(size_bytes, numa_node_);
        if (!p) {
            throw Error("numa_alloc_on_node failed");
        }
        // libnuma returns page-aligned memory; we accept that even if > alignment
        base_ = p;
        size_ = size_bytes;
        if (zero_init) {
            std::memset(base_, 0, size_);
        }
        return;
    }
#endif
    p = aligned_alloc_bytes(size_bytes, alignment_);
    if (!p) {
        throw Error("aligned allocation failed");
    }
    base_ = p;
    size_ = size_bytes;
    if (zero_init) {
        std::memset(base_, 0, size_);
    }
}

void* nnml_buffer::allocate_obj(size_t bytes) {
    NNML_ASSERT_MSG(base_ != nullptr, "Buffer not allocated yet!");

    size_t aligned_offset = (offset_ + alignment_ - 1) & ~(alignment_ - 1);
    if (aligned_offset + bytes > size_) {
        throw Error("Buffer OOM: not enough space for tensor allocation");
    }
    void* ptr = (uint8_t*)base_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void nnml_buffer::release() {
    if (!base_) return;
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    // We don't track whether allocation came from numa_alloc_on_node vs aligned_alloc.
    // Heuristic: if numa_node_ >= 0 we used numa_alloc_on_node.
    if (numa_node_ >= 0) {
        numa_free_on_node(base_, size_);
    } else {
        aligned_free_bytes(base_);
    }
#else
    aligned_free_bytes(base_);
#endif
    base_ = nullptr;
    size_ = 0;
    offset_ = 0;
    numa_node_ = -1;
    usage_ = NNML_BUFFER_USAGE_ANY;
}

void nnml_buffer::clear() {
    offset_ = 0;
}


// implementation of nnml_elastic_buffer
nnml_elastic_buffer::nnml_elastic_buffer(std::size_t size, bool zero_init, int numa_node, 
        nnml_buffer_usage usage, std::size_t alignment) {
    numa_node_ = numa_node;
    usage_ = usage;
    if ((alignment & (alignment - 1)) != 0) {
        throw Error("nnml_elastic_buffer::allocate(): alignment must be power of two");
    }
    alignment_ = std::max(alignment, nnml_get_page_size());
    // Initialize memory allocation logic
    allocate(size, zero_init);
}

void nnml_elastic_buffer::allocate(std::size_t size_bytes, bool zero_init) {
    buffer_region region;
    void* p = nullptr;
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
    if ((numa_node_ >= 0) && numa_is_available()) {
        p = numa_alloc_on_node(size_bytes, numa_node_);
        if (!p) {
            throw Error("numa_alloc_on_node failed");
        }
        // libnuma returns page-aligned memory; we accept that even if > alignment
        region.base_ = p;
        region.size_ = size_bytes;
        if (zero_init) {
            std::memset(region.base_, 0, region.size_);
        }
        regions_.emplace_back(std::move(region));
        return;
    }
#endif
    p = aligned_alloc_bytes(size_bytes, alignment_);
    if (!p) {
        throw Error("aligned allocation failed");
    }
    region.base_ = p;
    region.size_ = size_bytes;
    if (zero_init) {
        std::memset(region.base_, 0, region.size_);
    }
    regions_.emplace_back(std::move(region));
}

void* nnml_elastic_buffer::allocate_obj(size_t bytes) {
    NNML_ASSERT_MSG(n_regions() > 0, "No regions allocated yet!");

    std::size_t last_remaining = remaining(n_regions() - 1);
    if (bytes > last_remaining) {
        NNML_LOG_INFO("Elastic Buffer: allocating new region of size %zu bytes\n", bytes);
        if (n_regions() > NNML_MAX_GROWTH_TRIES) {
            throw Error("Elastic Buffer OOM: exceeded maximum growth tries");
        }
        allocate(NNML_ELASTIC_BUFFER_GROWTH_SIZE, false);  // WARNING: currently we always allocate a fixed size region upon growth, param-bytes is better
    }
    size_t aligned_offset = (regions_.back().offset_ + alignment_ - 1) & ~(alignment_ - 1);
    size_t region_size_ = regions_.back().size_;
    if (aligned_offset + bytes > region_size_) {
        throw Error("Elastic Buffer OOM: not enough space for tensor allocation even after growth");
    }
    void* ptr = (uint8_t*)regions_.back().base_ + aligned_offset;
    regions_.back().offset_ = aligned_offset + bytes;
    return ptr;
}

void nnml_elastic_buffer::release() {
    if (!n_regions()) return;
    for (auto& region : regions_) {
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
        if (numa_node_ >= 0) {
            numa_free_on_node(region.base_, region.size_);
        } else {
            aligned_free_bytes(region.base_);
        }
#else
        aligned_free_bytes(region.base_);
#endif
        region.base_ = nullptr;
        region.size_ = 0;
    }
    numa_node_ = -1;
    usage_ = NNML_BUFFER_USAGE_ANY;
}

void nnml_elastic_buffer::clear() {
    NNML_ASSERT_MSG(n_regions() > 0, "No regions allocated yet!");

    for (auto& region : regions_) {
        if (&region == &regions_.front()) {
            // keep first region
            continue;
        }
#if defined(__linux__) && defined(NNML_HAS_LIBNUMA)
        if (numa_node_ >= 0) {
            numa_free_on_node(region.base_, region.size_);
        } else {
            aligned_free_bytes(region.base_);
        }
#else
        aligned_free_bytes(region.base_);
#endif
        region.base_ = nullptr;
        region.size_ = 0;
    }
    regions_.resize(1);
    regions_.front().offset_ = 0;
}


void* nnml_elastic_buffer::region_base(unsigned idx) const {
    if (idx >= regions_.size()) {
        throw Error("nnml_elastic_buffer::region_base(): index out of range");
    }
    return regions_[idx].base_;
}

std::size_t nnml_elastic_buffer::region_size(unsigned idx) const {
    if (idx >= regions_.size()) {
        throw Error("nnml_elastic_buffer::region_size(): index out of range");
    }
    return regions_[idx].size_;
}

std::size_t nnml_elastic_buffer::region_offset(unsigned idx) const {
    if (idx >= regions_.size()) {
        throw Error("nnml_elastic_buffer::region_offset(): index out of range");
    }
    return regions_[idx].offset_;
}

std::size_t nnml_elastic_buffer::size() const {
    std::size_t total = 0;
    for (const auto& region : regions_) {
        total += region.size_;
    }
    return total;
}

std::size_t nnml_elastic_buffer::used_size() const {
    std::size_t total = 0;
    for (const auto& region : regions_) {
        total += region.offset_;
    }
    return total;
}


// implementation of nnml_buffer_view
void* nnml_buffer_view::allocate_obj(std::size_t bytes) {
    if (!base_) throw Error("nnml_buffer_view::allocate_obj(): null base");
    if (bytes == 0) return nullptr;
    std::size_t aligned_offset = (offset_ + alignment_ - 1) & ~(alignment_ - 1);
    if (aligned_offset + bytes > size_) {
        throw Error("nnml_buffer_view OOM: not enough space in view");
    }
    void* ptr = (uint8_t*)base_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void nnml_buffer_view::clear() {
    if (!base_) return;
    offset_ = 0;
}


// implementation of nnml_dual_buffer
nnml_dual_buffer::nnml_dual_buffer(nnml_buffer& parent, std::size_t size, std::size_t alignment)
    : alignment_(alignment) {
    // assert parent is instance of nnml_buffer or nnml_elastic_buffer (only for the 1-st region)
    // i.e., if parent is nnml_elastic_buffer, there should only be the region 0 in it
    if (typeid(parent) == typeid(nnml_elastic_buffer)) {
        nnml_elastic_buffer& ebuff = static_cast<nnml_elastic_buffer&>(parent);
        if (ebuff.n_regions() != 1) {
            throw Error("nnml_dual_buffer: parent elastic buffer must have exactly one region");
        }
    }
    void* p0 = parent.allocate_obj(size);
    void* p1 = parent.allocate_obj(size);
    views_[0] = nnml_buffer_view(p0, size, alignment_);
    views_[1] = nnml_buffer_view(p1, size, alignment_);
}

nnml_buffer_view& nnml_dual_buffer::view(unsigned idx) {
    if (idx > 1) throw Error("nnml_dual_buffer: index out of range");
    return views_[idx];
}

const nnml_buffer_view& nnml_dual_buffer::view(unsigned idx) const {
    if (idx > 1) throw Error("nnml_dual_buffer: index out of range");
    return views_[idx];
}

void* nnml_dual_buffer::allocate_obj_on(unsigned idx, std::size_t bytes) { // allocate on selected sub-buffer (0 or 1)
    return views_[idx].allocate_obj(bytes);
}

void nnml_dual_buffer::clear_on(unsigned idx) {         // clear selected sub-buffer (0 or 1)
    views_[idx].clear();
}


// implementation of nnml_memory
nnml_memory_t nnml_memory::create(const cparams& params) {
    nnml_memory_t mem(new nnml_memory());
    mem->numa_info_ = query_numa_info();
    mem->alignment_ = params.alignment;
    if (params.strategy == NNML_ALLOC_STRATEGY_NUMA && !mem->numa_info_.available) {
        NNML_LOG_WARN("nnml_memory::create(): requested NUMA allocation strategy but NUMA is not available, falling back to UMA\n");
    }

    // when want_numa is true, it means we want to use NUMA distributed allocation
    const bool want_numa = (params.strategy == NNML_ALLOC_STRATEGY_NUMA) && mem->numa_info_.available;
    unsigned nodes = want_numa ? mem->numa_info_.node_count : 1;
    if (nodes > 1 && params.n_nodes_manual > 0) nodes = params.n_nodes_manual;
    int local_node_id = nnml_get_boot_node_id();
    if(!want_numa && mem->numa_info_.available && local_node_id < 0) {
        fprintf(stderr, "UMA in NUMA platform: Failed to get local NUMA node id and it will use node 0 for allocation.\n");
        local_node_id = 0;
    }

    std::size_t wsize = params.total_wsize_bytes / nodes;
    std::size_t csize = params.total_csize_bytes / nodes;
    std::size_t kvsize = params.total_kvsize_bytes / nodes;
    std::size_t tmp_work_size = params.tmp_work_size_bytes / nodes;
    
    for (unsigned i = 0; i < nodes; ++i) {
        unsigned alloc_node = want_numa ? i : (mem->numa_info_.available ? local_node_id : -1);
        printf("Allocating buffer on node: %u\n", alloc_node);
        nnml_buffer* wbuf = new nnml_buffer(wsize, params.zero_init, alloc_node,
                         NNML_BUFFER_USAGE_WEIGHTS, params.alignment);
        nnml_buffer* cbuf = new nnml_buffer(csize, params.zero_init, alloc_node,
                         NNML_BUFFER_USAGE_COMPUTE, params.alignment);
        nnml_buffer* tmp_work_buf = new nnml_buffer(tmp_work_size, params.zero_init, alloc_node,
                         NNML_BUFFER_USAGE_COMPUTE, params.alignment);
        if (params.is_kv_elastic) {
            nnml_elastic_buffer* kvbuf = new nnml_elastic_buffer(kvsize, params.zero_init, alloc_node,
                         NNML_BUFFER_USAGE_COMPUTE, params.alignment);
            mem->buffers_.emplace_back(node_buffers(wbuf, cbuf, kvbuf, tmp_work_buf));
        } else {
            nnml_buffer* kvbuf = new nnml_buffer(kvsize, params.zero_init, alloc_node,
                         NNML_BUFFER_USAGE_COMPUTE, params.alignment);
            mem->buffers_.emplace_back(node_buffers(wbuf, cbuf, kvbuf, tmp_work_buf));
        }
    }
    
    return mem;
}

int nnml_memory::create_dual_buffers(std::size_t single_size) {
    dual_buffers_.clear();
    for (auto& node_buf : buffers_) {
        nnml_buffer& abuf = *(node_buf.a_buffer);
        nnml_dual_buffer dual_buf(abuf, single_size, alignment_);
        dual_buffers_.emplace_back(std::move(dual_buf));
    }
    return 0;
}

void nnml_memory::release() {
    // first release dual buffers
    dual_buffers_.clear();
    // then release all buffers
    for (auto& node_buf : buffers_) {
        node_buf.w_buffer->release();
        node_buf.a_buffer->release();
        node_buf.kv_buffer->release();
        node_buf.tmp_work_buffer->release();
    }
}

std::size_t nnml_memory::total_weight_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.w_buffer->size();
    return sum;
}

std::size_t nnml_memory::total_activation_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.a_buffer->size();
    return sum;
}

std::size_t nnml_memory::total_kv_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.kv_buffer->size();
    return sum;
}

std::size_t nnml_memory::total_size() const {
    return total_weight_size() + total_activation_size() + total_kv_size();
}

std::size_t nnml_memory::total_used_weight_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.w_buffer->used_size();
    return sum;
}

std::size_t nnml_memory::total_used_activation_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.a_buffer->used_size();
    return sum;
}

std::size_t nnml_memory::total_used_kv_size() const {
    std::size_t sum = 0;
    for (const auto& node_buf : buffers_) sum += node_buf.kv_buffer->used_size();
    return sum;
}

std::size_t nnml_memory::total_used_size() const {
    return total_used_weight_size() + total_used_activation_size() + total_used_kv_size();
}

nnml_buffer_t nnml_memory::weight_buffer(unsigned node) {
    if (node >= buffers_.size()) {
        throw Error("nnml_memory::weight_buffer(): node index out of range");
    }
    return buffers_[node].w_buffer;
}

nnml_buffer_t nnml_memory::activation_buffer(unsigned node) {
    if (node >= buffers_.size()) {
        throw Error("nnml_memory::activation_buffer(): node index out of range");
    }
    return buffers_[node].a_buffer;
}

nnml_buffer_t nnml_memory::kv_buffer(unsigned node) {
    if (node >= buffers_.size()) {
        throw Error("nnml_memory::kv_buffer(): node index out of range");
    }
    return buffers_[node].kv_buffer;
}

nnml_buffer_t nnml_memory::tmp_work_buffer(unsigned node) {
    if (node >= buffers_.size()) {
        throw Error("nnml_memory::tmp_work_buffer(): node index out of range");
    }
    return buffers_[node].tmp_work_buffer;
}

nnml_dual_buffer& nnml_memory::dual_buffer(unsigned node) {
    if (dual_buffers_.empty()) {
        throw Error("nnml_memory::dual_buffer(): dual buffers not built");
    }
    if (node >= dual_buffers_.size()) {
        throw Error("nnml_memory::dual_buffer(): node index out of range");
    }
    return dual_buffers_[node];
}

nnml_buffer_t nnml_memory::single_weight_buffer() {
    if (buffers_.size() != 1) {
        throw Error("nnml_memory::single_weight_buffer(): not a UMA memory (buffers.size != 1)");
    }
    return buffers_[0].w_buffer;
}

nnml_buffer_t nnml_memory::single_activation_buffer() {
    if (buffers_.size() != 1) {
        throw Error("nnml_memory::single_activation_buffer(): not a UMA memory (buffers.size != 1)");
    }
    return buffers_[0].a_buffer;
}

nnml_buffer_t nnml_memory::single_tmp_work_buffer() {
    if (buffers_.size() != 1) {
        throw Error("nnml_memory::single_tmp_work_buffer(): not a UMA memory (buffers.size != 1)");
    }
    return buffers_[0].tmp_work_buffer;
}

void* nnml_memory::allocate_obj(nnml_tensor_type tensor_type, int32_t buffer_id, int32_t dual_idx, size_t bytes) {
    NNML_ASSERT_MSG(buffers_.size() > 0, "No buffers available in memory!");
    if (tensor_type == NNML_TENSOR_TYPE_WEIGHT) {
        if (buffer_id < 0 || buffer_id >= buffers_.size()) {
            throw Error("nnml_memory::allocate_obj(): weight buffer_id out of range");
        }
        return buffers_[buffer_id].w_buffer->allocate_obj(bytes);
    } else if (tensor_type == NNML_TENSOR_TYPE_KVCACHE) {
        if (buffer_id < 0 || buffer_id >= buffers_.size()) {
            throw Error("nnml_memory::allocate_obj(): kv_cache buffer_id out of range");
        }
        return buffers_[buffer_id].kv_buffer->allocate_obj(bytes);
    } else if (tensor_type == NNML_TENSOR_TYPE_ACTIVATION) {
        if (buffer_id < 0 || buffer_id >= buffers_.size()) {
            throw Error("nnml_memory::allocate_obj(): activation buffer_id out of range");
        }
        return buffers_[buffer_id].a_buffer->allocate_obj(bytes);
    } else if (tensor_type == NNML_TENSOR_TYPE_DUAL_ACTI) {
        NNML_ASSERT_MSG(dual_idx == 0 || dual_idx == 1, "dual_idx must be 0 or 1");
        if (dual_buffers_.empty()) {
            throw Error("nnml_memory::allocate_obj(): dual buffers not built");
        }
        if (buffer_id < 0 || buffer_id >= buffers_.size()) {
            throw Error("nnml_memory::allocate_obj(): dual buffer_id out of range");
        }
        return dual_buffers_[buffer_id].allocate_obj_on(dual_idx, bytes);
    } else {
        throw Error("nnml_memory::allocate_obj(): unknown tensor_type");
        return nullptr;
    }
}

int nnml_memory::clear_activation_buffers() {
    dual_buffers_.clear();
    for (auto& node_buf : buffers_) {
        node_buf.a_buffer->clear();
    }
    return 0;
}

int nnml_memory::clear_kv_buffers() {
    for (auto& node_buf : buffers_) {
        node_buf.kv_buffer->clear();
    }
    return 0;
}
