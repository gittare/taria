#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <mutex>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    }

namespace taria {
namespace runtime {

/// Represents a CUDA stream for async execution
class Stream {
public:
    Stream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
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

/// A simple thread-safe pinned memory allocator/pool concept
class PinnedAllocator {
public:
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        void* ptr;
        CUDA_CHECK(cudaMallocHost(&ptr, size));
        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        CUDA_CHECK(cudaFreeHost(ptr));
    }

private:
    std::mutex mutex_;
};

/// High-level scheduler for submitting kernel tasks to streams
class Scheduler {
public:
    Scheduler(int num_streams) {
        for (int i = 0; i < num_streams; ++i) {
            streams_.emplace_back(new Stream());
        }
    }

    Stream* get_stream(int index) {
        if (index < 0 || index >= streams_.size()) {
            return nullptr;
        }
        return streams_[index].get();
    }

    void wait_all() {
        for (auto& s : streams_) {
            s->synchronize();
        }
    }

private:
    std::vector<std::unique_ptr<Stream>> streams_;
};

} // namespace runtime
} // namespace taria

extern "C" {
    // C API for FFI integration with Rust
    void* taria_stream_create() {
        return new taria::runtime::Stream();
    }

    void taria_stream_sync(void* stream_ptr) {
        auto* stream = static_cast<taria::runtime::Stream*>(stream_ptr);
        stream->synchronize();
    }

    void taria_stream_destroy(void* stream_ptr) {
        auto* stream = static_cast<taria::runtime::Stream*>(stream_ptr);
        delete stream;
    }
}
