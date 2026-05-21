# Semantic Compression & Async Runtime Lowering

This document traces the exact path a Taria semantic compression script takes through the backend architecture, demonstrating the seamless integration between high-level language constructs and low-level CUDA runtime mechanisms.

## 1. Lowering a Compression Kernel

Consider the following Taria source:
```taria
@gpu.kernel(block_size=256)
fn compress_chunk(chunk: Tensor<f16>) -> QuantizedTensor {
    latent = encoder(chunk)
    quantized = latent.quantize(bits=4)
    return quantized
}
```

### AST -> MLIR Translation
The `ASTToTaria` C++ pass maps this to:
```mlir
taria.gpu.kernel block_size(256) {
    %latent = taria.compress.encode %chunk : (!taria.tensor<...>) -> !taria.latent<...>
    %quant = taria.compress.quantize %latent {bits = 4} : (!taria.latent<...>) -> !taria.quantized<...>
    taria.return %quant
}
```

### Fusion and Linalg Mapping
The `TensorFusionEngine` sees `%latent` is consumed immediately and never returned. It fuses the operations. The `TariaToLinalg` pass then lowers the encoding into a sequence of affine generic loops, and inserts the quantization math directly into the innermost loop body.

### PTX Emission
The resulting LLVM IR is fed into the PTX emitter. The compiler generates an `fma` (fused multiply-add) loop for the neural encoder. At the end of the loop, instead of storing the result, the value in the floating-point register is compared against the codebook (pre-loaded in `__shared__` memory) using branchless instructions, and only the 4-bit index is stored to global memory.

## 2. Async Runtime Lowering

Taria enables overlapping compute and IO natively.

```taria
async pipeline semantic_stream {
    chunk = input.read()
    output.write(encoder(chunk))
}
```

### Stream Management Lowering
The `AsyncScheduler` pass translates the `async pipeline` block into a state machine controlled by the `CudaScheduler` C++ API.
- The `.read()` operation compiles to a PCIe DMA transfer mapped to CUDA Stream 0 (Copy Engine).
- The `encoder()` compiles to a Kernel Launch mapped to CUDA Stream 1 (Compute Engine).
- A `cudaEvent_t` is injected between them. Stream 1 is instructed to `cudaStreamWaitEvent`, allowing the CPU to return immediately and queue the next chunk of data while the GPU executes autonomously.

## 3. Distributed Backend (Roadmap)

To handle models or tensors that exceed a single GPU's VRAM, Taria is adding distributed support via the `distributed/` backend module.

### Lowering `@sharded` Tensors
If a tensor is declared as `chunk: DistributedTensor<f32, Shards=8>`, the semantic IR flags this for the `DistributedScheduler`.
- The MLIR lowering pass inserts `taria.dist.all_gather` or `taria.dist.reduce_scatter` nodes into the compute graph.
- During runtime, these operations map directly to the NVIDIA NCCL C++ API.
- By inserting these at the MLIR level, the `AsyncScheduler` can overlap NCCL network communication with neural compute, ensuring the NVLink/NVSwitch interconnects are fully saturated simultaneously with the SMs.
