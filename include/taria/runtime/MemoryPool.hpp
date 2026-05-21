#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace taria {
namespace runtime {

/// High-performance Memory Pool for Zero-Copy and Unified Memory Allocations.
///
/// Purpose:
/// AI-native compression involves shuffling massive (100GB+) tensors. Traditional
/// `cudaMalloc` and `cudaFree` involve heavy synchronization overhead. This pool
/// caches Pinned (HostAllocMapped) and Device memory.
///
/// Threading Model:
/// Thread-safe via internal mutexes. Can be accessed concurrently by multiple
/// async CUDA stream dispatchers.
class MemoryPool {
public:
    static MemoryPool& get_instance();

    /// Allocates page-locked host memory mapped into device address space.
    void* allocate_pinned(size_t size_bytes);

    /// Frees pinned memory back into the cache.
    void free_pinned(void* ptr);

    /// Allocates direct VRAM.
    void* allocate_device(size_t size_bytes);

    /// Frees direct VRAM back into the cache.
    void free_device(void* ptr);

    /// Flushes all cached blocks to the OS/Driver.
    void clear_cache();

private:
    MemoryPool() = default;
    ~MemoryPool();

    std::mutex pinned_mutex_;
    std::mutex device_mutex_;

    // Cache structures mapping block sizes to available pointers
    std::unordered_map<size_t, std::vector<void*>> free_pinned_blocks_;
    std::unordered_map<size_t, std::vector<void*>> free_device_blocks_;

    // Tracking allocation sizes for free() lookup
    std::unordered_map<void*, size_t> active_pinned_;
    std::unordered_map<void*, size_t> active_device_;
};

} // namespace runtime
} // namespace taria
