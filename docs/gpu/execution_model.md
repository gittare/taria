# Taria GPU Execution & Optimization Model

The Taria compiler is built specifically to extract maximum performance from modern GPUs (NVIDIA Ampere/Hopper, AMD RDNA/CDNA) when executing massive tensor operations for semantic compression.

## Thread / Block Mapping & Occupancy

Taria maps operations defined in the `taria` MLIR dialect into the standard `gpu.launch` dialect.

- **Warp Execution:** Threads execute in groups of 32 (warps). Divergent branching inside a warp causes serialization. Taria's standard library implements vector quantization and codebook lookups using branchless SIMD primitives (e.g., `__shfl_sync` and bitwise selects) to guarantee warps execute identically.
- **Occupancy:** Defined as the ratio of active warps on an SM to the maximum possible warps. Taria allows developers to explicitly decorate kernels with `@gpu.kernel(block_size=256, shared_mem="32kb")`. The MLIR lowering pass statically calculates register pressure and shared memory limits to ensure maximum occupancy.

## Memory Hierarchy and Coalescing

Moving terabytes of data requires stringent memory engineering.

### Global Memory (VRAM) -> Shared Memory (SRAM)

Taria's `LinalgToGPU` pass automatically promotes highly accessed tensors (like learned quantization codebooks) into Shared Memory (`__shared__` in PTX).
- By staging codebooks in SRAM, we drop lookup latencies from ~300 cycles (Global) to ~30 cycles (Shared).

### Memory Coalescing

Taria enforces memory alignment in the Rust semantic analyzer.
- Contiguous threads in a warp must access contiguous memory addresses.
- If a tensor's innermost dimension is not a multiple of 32 (or 16 for `f16`), the compiler automatically inserts padding (stride modifications) during MLIR lowering to ensure 128-byte aligned memory transactions.

## Kernel Fusion

Traditional ML frameworks execute compression pipelines sequentially:
1. Load data, encode, store latent.
2. Load latent, quantize, store discrete values.

This is bottlenecked by global memory bandwidth. Taria leverages MLIR to apply aggressive **Kernel Fusion**.

The pass pipeline pattern matches sequential `taria.encode` and `taria.vector_quantize` ops. It emits a single PTX kernel that:
1. Loads raw data from VRAM.
2. Computes the latent vector in registers.
3. Quantizes the vector against an SRAM-cached codebook.
4. Writes *only* the compressed output back to VRAM.

## Asynchronous Pipelines

The C++ Runtime (`runtime/taria_runtime.cu`) utilizes CUDA Streams (`cudaStream_t`) to execute Taria kernels asynchronously.
- Memory transfers from the host to the GPU use zero-copy pinned memory (`cudaHostAllocMapped`).
- The scheduler queues transfers and kernels concurrently, ensuring the GPU's compute engines and copy engines are fully saturated.
