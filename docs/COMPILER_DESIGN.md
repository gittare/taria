# Taria Compiler Architecture & Engineering Guidelines

This document serves as the comprehensive engineering guide for the Taria compiler, aimed at elite compiler engineers, systems researchers, and MLIR/LLVM contributors.

## 1. Architectural Philosophy

Taria bridges the gap between ultra-high-level AI semantics (autoencoders, entropy modeling) and low-level hardware execution (NVPTX, AMDGCN, SPIR-V). The core philosophy relies on **Semantic Preservation**. By maintaining domain-specific constructs inside the `taria` MLIR dialect, the compiler can make global optimization decisions (like fusing quantization with entropy encoding) before shredding the representation into Linalg generics.

### Key Goals
- **Extreme Latent Compression**: Drive terabytes of data down to gigabytes by moving neural compression natively into the compiler graph.
- **Zero-Cost Abstractions**: Rust’s arena-allocated AST parsing incurs absolutely zero runtime penalty in the generated PTX assembly.
- **Occupancy Maximization**: Smart block-size inference and automatic shared memory layout tiling.

---

## 2. The Hybrid Compilation Pipeline

### Phase 1: Rust Frontend (`frontend/`)
The parser enforces type constraints and tensor shapes during Semantic Analysis (`sema.rs`).
- **Type Checking:** Ensures data flows correctly between neural encoders (`Tensor[f32]`) and quantizers (`Tensor[i8]`).
- **AST Generation:** An immutable, arena-allocated AST is generated.

### Phase 2: MLIR Lowering (`passes/`)
1. **`ASTToTaria`**: Transcribes the Rust AST to the `taria` dialect.
2. **`Taria Optimization`**:
   - **Tensor Fusion**: Pattern rewriters match sequences like `taria.encode -> taria.vector_quantize` and fuse them to avoid global memory round-trips.
   - **Memory Coalescing**: Adjusts tensor layouts to align with warp boundaries (e.g., ensuring 128-byte aligned contiguous loads).
3. **`TariaToLinalg`**: Lowers macro operations into affine loops and generic tensor compute blocks.
4. **`LinalgToGPU`**: Translates loops into `gpu.launch`, mapping `linalg` parallel loops to blocks and threads.

### Phase 3: Hardware Target (`backend/`)
1. **`GPUToNVVM/AMDGCN`**: Lowers standard GPU ops to architecture-specific LLVM IR dialects.
2. **LLVM CodeGen**: LLVM passes run (e.g., loop unrolling, instruction scheduling) to emit the final device assembly (PTX or ISA).

---

## 3. GPU Memory Optimization Strategy

To sustain high memory bandwidth during compression:

### Tensor Tiling & Alignment (`taria.tensor_tile`)
Taria automatically applies explicit tiling maps (`taria.tensor_tile`) based on the target hardware's L1/L2 cache sizes.
- Shared Memory (`__shared__`) is aggressively utilized by staging latent tensors into SRAM before entropy modeling.
- Bank conflicts are mitigated by automatic padding inserted during the `LinalgToGPU` pass.

### Warp Divergence Mitigation
Neural compression inherently involves branches (e.g., finding the nearest neighbor in a codebook). Taria's standard library implements these using branchless SIMD primitives (e.g., `__shfl_sync` / `select` instructions in LLVM IR) to guarantee all threads in a warp execute synchronously.

---

## 4. Asynchronous Execution (`runtime/`)

The Taria runtime (`taria_runtime.cu`) implements a high-performance C++ task scheduler.
- **Pinned Memory (`cudaMallocHost`)**: Tensors transferred between host and device use pinned memory pools to maximize PCIe bandwidth.
- **CUDA Streams**: Execution kernels are dispatched asynchronously onto independent hardware queues.
- **Graph Capture (Future)**: Support for CUDA Graphs to eliminate CPU submission overhead during iterative compression cycles.

---

## 5. Extensibility: JIT & Distributed Environments

- **JIT Compilation**: While Taria defaults to AOT, the MLIR ExecutionEngine is built into the backend to allow dynamic JIT compilation for varying tensor shapes at runtime.
- **Distributed Tensors**: Future iterations of `sema.rs` will include `ShardedTensor` types, automatically emitting NCCL `all_reduce` or `all_gather` nodes into the MLIR graph to support compression across multi-GPU rings.
