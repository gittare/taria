# Taria MLIR Dialect & Type System

The `taria` MLIR dialect is the connective tissue between Taria's AI-native syntax and LLVM's rigorous code generation. It maintains the semantic intent of compression and tensor operations to allow graph-level optimizations before shredding down to loops.

## 1. Backend-Aware Type System

Taria introduces custom MLIR types to strictly model hardware layout and compression state.

- **`!taria.tensor<shape, type, layout>`**: Maps to standard tensors but tracks layout metadata (e.g., `DenseRowMajor`, `TiledZCurve`) to guide the `MemoryOptimizer` during bufferization.
- **`!taria.latent<shape, type>`**: Denotes a continuous vector space resulting from an autoencoder. Signals the `TensorFusionEngine` that this data should reside in registers or SRAM, avoiding VRAM writes if a `quantize` operation immediately follows.
- **`!taria.quantized<shape, bits>`**: Represents discrete indices. Lowering guarantees packed memory representations (e.g., packing two 4-bit values into one `i8` during VRAM stores).
- **`!taria.async<T>`**: A future/promise type mapping to a CUDA Stream Event.

## 2. Operation Definitions

### `taria.compress.encode`
```mlir
%latent = taria.compress.encode %chunk : (!taria.tensor<...>) -> !taria.latent<...>
```
- **Semantics**: Executes a neural encoder block.
- **Lowering Behavior**: Expands into a sequence of `linalg.conv_2d` and `linalg.matmul` ops.
- **GPU Mapping**: If immediately consumed, outputs are yielded directly into the `linalg.generic` register file.

### `taria.compress.quantize`
```mlir
%quant, %indices = taria.compress.quantize %latent, %codebook : (!taria.latent<...>, !taria.tensor<...>) -> (!taria.quantized<...>)
```
- **Semantics**: Nearest-neighbor vector quantization.
- **Optimization**: The `codebook` operand is tagged for `SharedMemoryPromotion`. During the `LinalgToGPU` pass, `gpu.subgroup_mma` or explicit `__shared__` allocation instructions are inserted to cache the codebook.
- **GPU Mapping**: Lowers to branchless `select` and `__shfl_sync` PTX instructions to eliminate warp divergence during the search loop.

### `taria.async.pipeline`
```mlir
%token = taria.async.pipeline {
   // blocks of compute
}
```
- **Semantics**: Defines an overlapping compute/memory boundary.
- **Lowering Behavior**: Transforms into `async.execute` and ultimately into `gpu.launch` attached to non-default `cudaStream_t` pointers, accompanied by `cudaEventRecord` for dependency tracking.

### `taria.tensor.fuse`
```mlir
%fused = taria.tensor.fuse {
   %0 = taria.mul %a, %b
   %1 = taria.add %0, %c
   taria.yield %1
}
```
- **Semantics**: An explicit or compiler-inferred region of operations guaranteed to be executed within a single kernel launch.
- **Optimization**: The contents of the region bypass VRAM bufferization, relying entirely on register allocation (PTX `%f1`, `%f2`).

## 3. Shape Inference and Verification

Every operation in the `taria` dialect implements the `InferTypeOpInterface`.
- **Verification**: MLIR strictly validates that a `!taria.latent` passed into `taria.compress.quantize` matches the dimensionality of the provided codebook.
- **Compile-time Analysis**: Strides and memory footprints are calculated statically to fail compilation if a requested `@gpu.kernel(shared_memory=...)` exceeds the hardware's SM capacity (e.g., > 164KB on Ampere).
