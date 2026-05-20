#pragma once

#include "taria/runtime/CudaScheduler.hpp"
#include "taria/runtime/MemoryPool.hpp"
#include <memory>

namespace taria {
namespace runtime {

/// Central initialization and configuration object for the Taria execution environment.
///
/// Purpose:
/// Bootstraps CUDA contexts, initializes Memory Pools, and configures the MLIR
/// ExecutionEngine (JIT) if running in dynamic mode.
///
/// Threading Model:
/// Initialization is not thread-safe. It must be called once per process boundary.
class GpuRuntime {
public:
    static GpuRuntime& get();

    void initialize();
    void shutdown();

    CudaScheduler& scheduler() { return *scheduler_; }
    MemoryPool& memory() { return MemoryPool::get_instance(); }

private:
    GpuRuntime() = default;
    ~GpuRuntime() = default;

    std::unique_ptr<CudaScheduler> scheduler_;
    bool initialized_ = false;
};

} // namespace runtime
} // namespace taria
