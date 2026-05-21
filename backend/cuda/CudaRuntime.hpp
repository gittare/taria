#pragma once

#include <memory>
#include <vector>

namespace taria {
namespace cuda {

// Forward declarations mapping to our include/taria API headers
namespace runtime { class HardwareStream; }

/// CudaRuntime acts as the low-level bridge between the API layer and the OS Driver.
///
/// Purpose:
/// Manages CUDA device contexts, primary execution streams, and graph capture APIs.
/// This object is owned by the global `GpuRuntime` singleton.
class CudaRuntime {
public:
    static CudaRuntime& get_instance();

    void initialize_device(int device_id = 0);

    /// Returns the number of available streaming multiprocessors (SMs)
    int get_sm_count() const;

    /// Returns the maximum amount of Shared Memory (SRAM) per block in bytes
    size_t get_max_shared_memory_per_block() const;

private:
    CudaRuntime() = default;
    int device_id_ = 0;
    bool initialized_ = false;
};

} // namespace cuda
} // namespace taria
