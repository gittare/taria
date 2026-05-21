#include "CudaRuntime.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace taria {
namespace cuda {

CudaRuntime& CudaRuntime::get_instance() {
    static CudaRuntime instance;
    return instance;
}

void CudaRuntime::initialize_device(int device_id) {
    if (initialized_) return;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA device.");
    }

    device_id_ = device_id;
    initialized_ = true;
}

int CudaRuntime::get_sm_count() const {
    if (!initialized_) return 0;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id_);
    return sm_count;
}

size_t CudaRuntime::get_max_shared_memory_per_block() const {
    if (!initialized_) return 0;
    int shared_mem;
    // Query max dynamic shared memory capacity for Hopper/Ampere
    cudaDeviceGetAttribute(&shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id_);
    return static_cast<size_t>(shared_mem);
}

} // namespace cuda
} // namespace taria
