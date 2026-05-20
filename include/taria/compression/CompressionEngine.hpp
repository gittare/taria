#pragma once

#include "taria/tensor/Tensor.hpp"
#include <memory>

namespace taria {
namespace compression {

/// Represents a deeply compressed neural representation of a Tensor.
///
/// Purpose:
/// Serves as the typed output of `taria.encode`. Latent vectors live in
/// continuous space before being passed into vector quantization.
///
/// Memory Layout:
/// Layout is heavily optimized for target hardware warp-sizes. Instead of NHWC or NCHW,
/// it utilizes `TiledZCurve` layout internally to maximize SRAM cache hits during
/// spatial entropy modeling.
class LatentVector {
public:
    explicit LatentVector(std::unique_ptr<tensor::Tensor> data);

    const tensor::Tensor& data() const { return *tensor_data_; }
    tensor::Tensor& data() { return *tensor_data_; }

    size_t latent_channels() const;
    size_t spatial_size() const;

private:
    std::unique_ptr<tensor::Tensor> tensor_data_;
};

/// High-level API for Semantic Compression Orchestration.
///
/// Purpose:
/// While the MLIR compiler fuses these operations in the PTX, this C++ API
/// provides the entry points for the host runtime to invoke the pre-compiled
/// compression kernels.
class CompressionEngine {
public:
    CompressionEngine() = default;

    /// Runs the neural encoder and returns the continuous latent representation.
    LatentVector encode(const tensor::Tensor& input);

    /// Decodes a latent representation back into the target tensor space.
    std::unique_ptr<tensor::Tensor> decode(const LatentVector& latent);

    /// Quantizes a continuous latent vector into a discrete codebook index tensor.
    std::unique_ptr<tensor::Tensor> vector_quantize(const LatentVector& latent);

    /// Applies ANS (Asymmetric Numeral Systems) to compress indices into a bitstream.
    std::vector<uint8_t> entropy_compress(const tensor::Tensor& quantized_indices);
};

} // namespace compression
} // namespace taria
