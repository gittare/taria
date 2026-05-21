#pragma once

#include <vector>
#include <cstdint>

namespace taria {

// Forward declarations
namespace tensor { class Tensor; }
namespace compression { class LatentVector; }

namespace backend {
namespace compression {

/// Semantic Compressor Core Engine
///
/// Translates heavily optimized latent-space representations into quantized bitstreams.
/// Operates natively on Tensors previously buffered into the `MemoryPool`.
class SemanticCompressor {
public:
    SemanticCompressor();

    /// Applies a learned neural encoder to raw tensor data.
    /// In a fully JIT'd environment, this would invoke an ExecutionEngine.
    /// Here it stubs the runtime execution logic mapping.
    taria::compression::LatentVector execute_encoder(const taria::tensor::Tensor& chunk);

    /// Dispatches the Vector Quantization nearest-neighbor search kernel.
    std::vector<uint8_t> execute_quantization(const taria::compression::LatentVector& latent);

private:
    void* loaded_ptx_module_ = nullptr;
};

} // namespace compression
} // namespace backend
} // namespace taria
