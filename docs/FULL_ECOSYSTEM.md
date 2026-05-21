# Taria Full Ecosystem Architecture

Taria is engineered as a world-class AI compiler ecosystem on par with LLVM, TVM, Triton, and XLA. This document outlines the holistic architecture spanning the entire monorepo, delineating library ownership, compiler flows, runtime environments, and tensor optimization pipelines.

## 1. Monorepo Structure & Library Ownership

The repository is modularized into distinct domains, built around a strict Rust/C++ separation of concerns, orchestrated via Cargo and CMake.

### Frontend Systems (`Rust`)
- **`frontend/`**: The main Rust crate. Contains the zero-copy, SIMD-aware `lexer`, the Pratt-driven `parser`, an immutable arena-allocated `ast`, and the `semantic` analyzer.
- **`ffi/`**: The ABI-stable C interface exporting the AST to the C++ backend. Ensures panic isolation via `catch_unwind`.
- **`cli/` & `tools/`**: Contains `tariac` (the driver), diagnostics inspectors, and AST dump utilities.

### Backend Systems (`C++ / MLIR`)
- **`mlir/` & `dialects/`**: Defines the `taria` MLIR dialect (e.g., `taria.compress`, `taria.quantize`) via TableGen (ODS). Handles type inference and canonicalization.
- **`optimizer/`**: Central hub for graph rewrite engines. Executes pattern matchers (e.g., fusing `encode` + `quantize` into a single operation).
- **`nvptx/` & `llvm/`**: Manages the lowering of standard `gpu` dialects into LLVM IR and triggers PTX codegen.

### Execution Systems (`CUDA / C++`)
- **`runtime/`**: The async executor (`CudaScheduler`, `GpuRuntime`).
- **`memory/`**: Advanced zero-copy memory allocation (`MemoryPool`) for handling unified and pinned tensor data.
- **`tensor/`**: Hardware-agnostic metadata wrappers outlining shape, stride, and `Layout`.
- **`compression/`**: The high-level API orchestrating the invocation of compiled Latent-space encoders and Entropy models.

### Advanced Infrastructure
- **`distributed/`**: Sharding logic mapped onto NCCL rings for multi-GPU scaling.
- **`profiling/`**: Nsight/NVProf integration endpoints for tracing occupancy and stream utilization.

---

## 2. MLIR/LLVM/CUDA Integration Pipeline

Taria uses a strictly phased compilation model to ensure performance constraints are met before code hits the hardware.

1. **AST Injection**: The Rust AST is parsed by `ASTToTaria` and converted to `taria` MLIR dialect nodes.
2. **Semantic Fusion**: The `optimizer/` matches compression pipelines. E.g., `taria.encode -> taria.vector_quantize` is fused to avoid global memory round trips.
3. **Linalg Lowering**: The `taria` operations are translated into affine parallel loops using standard MLIR `linalg` operations.
4. **GPU Mapping**: Loops are mapped to Grid/Block execution dimensions. `taria.tensor_tile` annotations are used to push latency-critical latency data into `__shared__` memory via `gpu.subgroup_mma`.
5. **PTX Emission**: The IR is passed to the LLVM TargetMachine, emitting hardware-specific PTX assembly.

---

## 3. Performance Engineering Strategies

### Warp Divergence Reduction
Codebook lookups and entropy encoding fundamentally involve branching. Taria restricts divergent control flow in its internal libraries, implementing branchless search mechanisms via bitwise `select` operations and `__shfl_sync` cross-warp reductions, ensuring all threads in a warp step in unison.

### Memory Coalescing & Tensor Tiling
High-bandwidth tensor operations require 128-byte aligned transactions. The `tensor/` layout system forces `f32` and `f16` strides to align with 32-thread boundaries.
During MLIR `LinalgToGPU` lowering, `taria.tensor_tile` ensures 2D blocks of tensors are loaded contiguously into Shared Memory (SRAM) before processing, avoiding Bank Conflicts.

### Asynchronous Pipelining & Occupancy
The `runtime/` system does not use synchronous `cudaDeviceSynchronize`. It overlaps computation with memory transfers using multi-stream `CudaScheduler` objects. Taria ensures the PTX register count stays below hardware limits per-thread to guarantee maximum SM occupancy (often hitting 100% active warps).

---

## 4. Future Scalability

- **JIT & AOT Engine**: `jit/` will embed the `MLIRExecutionEngine` to compile custom network shapes at runtime (similar to PyTorch Inductor).
- **Universal Hardware (`WebGPU`, `TPU`, `ROCm`)**: The backend is architected so that the `Linalg` dialect can be lowered to `AMDGCN` for ROCm or `SPIR-V` for Vulkan, breaking free of exclusive NVIDIA lock-in.
- **AI-Guided Auto-Tuning**: Future integration of neural cost models to predict the most efficient block sizes and memory layouts before lowering, shifting optimization from trial-and-error to learned heuristics.
