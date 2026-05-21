# Taria Semantic Compression System

Taria introduces the concept of Semantic Compression natively into the language. Instead of relying on external libraries to map floats to integers, Taria provides high-level types and operations that compile down to optimized MLIR compression dialects (`taria.encode`, `taria.quantize`, `taria.entropy_model`).

## 1. Latent Vectors & Embeddings

Taria distinguishes between raw data (`Tensor`) and semantic representations (`LatentVector`).

```taria
import taria.compress.neural

# A LatentVector carries specific layout optimizations (e.g., TiledZCurve)
# to ensure spatial locality during quantization passes.
fn extract_features(raw_audio: Tensor<f32>) -> LatentVector<f16> {
    model: neural.AutoEncoder = load_model("audio_codec.bin")

    # Ownership is consumed; raw_audio is dropped from VRAM
    latent = model.encode(raw_audio)
    return latent
}
```

## 2. Vector Quantization (VQ)

To compress continuous space to discrete codes, Taria implements Vector Quantization as a zero-cost abstraction over highly optimized shared-memory PTX kernels.

```taria
import taria.compress.vq

@gpu.kernel(block_size=256, shared_memory=true)
fn quantize_latents(latent: LatentVector<f16>, codebook: Tensor<f16, [1024, 64]>) -> QuantizedTensor<u16> {
    # Taria recognizes 'vq.nearest_neighbor' and automatically stages the 'codebook'
    # into SRAM (__shared__) to prevent catastrophic VRAM bottlenecks.
    indices = vq.nearest_neighbor(latent, codebook)
    return QuantizedTensor(indices)
}
```

## 3. Entropy-Aware Tensors

Once quantized, indices are compressed to the Shannon entropy limit using learned probability distributions.

```taria
import taria.compress.entropy

fn compress_to_disk(indices: QuantizedTensor<u16>, cdf: Tensor<f32>) -> Bitstream {
    # Applies Asymmetric Numeral Systems (ANS) directly on the GPU.
    # The output is a highly compressed byte array ready for network/disk transport.
    stream: Bitstream = entropy.ans_encode(indices, cdf)
    return stream
}
```

## 4. End-to-End Streaming Pipelines

Semantic compression shines when applied to real-time streaming data (e.g., video, spatial audio). Taria uses `async pipeline` constructs to manage the overlap between PCIe transfers, GPU kernel execution, and disk I/O.

```taria
# Defines a long-running, asynchronous GPU pipeline
async pipeline realtime_neural_codec(input_stream: Stream<Tensor<f32>>, out_disk: FileStream) {
    # Loop unrolling and software pipelining are handled by the compiler
    while mut chunk = input_stream.read_async() {

        # 1. Neural Encoding
        latent = model.encode(chunk)

        # 2. Vector Quantization
        indices = quantize_latents(latent, codebook)

        # 3. Entropy Coding
        bits = compress_to_disk(indices, learned_cdf)

        # 4. Asynchronous write back to host/disk
        out_disk.write_async(bits)
    }
}
```

### Compiler Semantics for Pipelines
When the Taria compiler encounters an `async pipeline`, it:
1. Allocates a `MemoryPool` using pinned memory (`cudaHostAllocMapped`).
2. Dispatches `read_async` and `write_async` to a dedicated CUDA Copy Engine stream.
3. Dispatches `model.encode` and `quantize` to a dedicated CUDA Compute Engine stream.
4. Inserts CUDA Events (`cudaEventRecord`) automatically to synchronize data boundaries without stalling the CPU.
