#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <unordered_map>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    }

namespace taria {
namespace runtime {

/// Advanced Asynchronous Execution Stream
class Stream {
public:
    Stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~Stream() {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t get() const { return stream_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_;
};

/// High-performance memory pool leveraging Unified Memory / Pinned Pages
/// designed for zero-copy PCIe transfers required by massive tensors.
class MemoryPool {
public:
    void* allocate_pinned(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        // If we have a cached chunk of sufficient size, return it
        for (auto it = free_pinned_.begin(); it != free_pinned_.end(); ++it) {
            if (it->second >= size) {
                void* ptr = it->first;
                free_pinned_.erase(it);
                return ptr;
            }
        }

        // Otherwise, allocate new pinned memory (HostAllocMapped for ZeroCopy)
        void* ptr;
        CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocMapped));
        allocated_sizes_[ptr] = size;
        return ptr;
    }

    void deallocate_pinned(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (allocated_sizes_.find(ptr) != allocated_sizes_.end()) {
            free_pinned_.push_back({ptr, allocated_sizes_[ptr]});
        }
    }

    ~MemoryPool() {
        for (const auto& pair : allocated_sizes_) {
            cudaFreeHost(pair.first);
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<void*, size_t> allocated_sizes_;
    std::vector<std::pair<void*, size_t>> free_pinned_;
};

/// High-level scheduler for dispatching Taria PTX closures
class Scheduler {
public:
    Scheduler(int num_streams) {
        for (int i = 0; i < num_streams; ++i) {
            streams_.emplace_back(new Stream());
        }
    }

    Stream* get_next_stream() {
        int idx = current_stream_idx_++ % streams_.size();
        return streams_[idx].get();
    }

    void wait_all() {
        for (auto& s : streams_) {
            s->synchronize();
        }
    }

private:
    std::vector<std::unique_ptr<Stream>> streams_;
    std::atomic<int> current_stream_idx_{0};
};

// Global Runtime State
static MemoryPool G_MEMORY_POOL;
static Scheduler* G_SCHEDULER = nullptr;

} // namespace runtime
} // namespace taria

extern "C" {
    void taria_runtime_init(int num_streams) {
        taria::runtime::G_SCHEDULER = new taria::runtime::Scheduler(num_streams);
    }

    void taria_runtime_shutdown() {
        delete taria::runtime::G_SCHEDULER;
    }

    void* taria_alloc_tensor_host(size_t size_bytes) {
        return taria::runtime::G_MEMORY_POOL.allocate_pinned(size_bytes);
    }

    void taria_free_tensor_host(void* ptr) {
        taria::runtime::G_MEMORY_POOL.deallocate_pinned(ptr);
    }
}
