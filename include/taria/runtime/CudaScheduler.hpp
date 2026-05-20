#pragma once

#include <memory>
#include <functional>
#include "taria/tensor/Tensor.hpp"

namespace taria {
namespace runtime {

/// Opaque wrapper around a native hardware stream (e.g., cudaStream_t).
class HardwareStream;

/// CudaScheduler manages asynchronous kernel execution pipelines.
///
/// Purpose:
/// Overlaps memory transfers with PTX kernel execution. Taria compiler emits
/// async dependency graphs that this scheduler translates into non-blocking
/// stream submissions.
///
/// Performance Implications:
/// Uses round-robin stream selection (or dependency-aware graph topological sorts)
/// to ensure the GPU copy engines and compute units (SMs) are maximally saturated.
class CudaScheduler {
public:
    explicit CudaScheduler(int pool_size = 4);
    ~CudaScheduler();

    /// Submits an asynchronous task to a non-blocking stream.
    void submit(std::function<void(HardwareStream*)> task);

    /// Records an event on the given stream for synchronization.
    void record_event(HardwareStream* stream, int event_id);

    /// Makes `stream` wait on `event_id` completing.
    void wait_event(HardwareStream* stream, int event_id);

    /// Blocks the host CPU until all streams have completed execution.
    void synchronize_all();

private:
    std::vector<std::unique_ptr<HardwareStream>> streams_;
    int next_stream_idx_ = 0;
};

} // namespace runtime
} // namespace taria
