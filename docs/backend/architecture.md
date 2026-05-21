# Taria Backend Architecture

The Taria backend is a world-class, production-grade compiler infrastructure built on top of MLIR and LLVM. It is explicitly engineered to translate the high-level semantic AI compression paradigms of the Taria DSL into ultra-high-performance GPU machine code.

## 1. Backend Compilation Pipeline

The pipeline is strictly phased to ensure maximum modularity and optimization capability at every abstraction level:

```text
Taria Source
      ↓
[Rust] AST & Semantic IR (Ownership, Types, Async boundaries mapped)
      ↓
[C++] Taria MLIR Dialect (High-level tensors, Latents, Compression Ops)
      ↓
[C++] Tensor Optimization Passes (Tensor Fusion, Tiling, Auto-tuning heuristics)
      ↓
[C++] Linalg Dialect (Generic tensor loops and parallel maps)
      ↓
[C++] GPU Dialect (Thread/Block mapping, Shared Memory allocation)
      ↓
[C++] NVVM Dialect (NVIDIA-specific intrinsics)
      ↓
[C++] LLVM IR (Standard LLVM optimization passes)
      ↓
[LLVM Target] PTX (Assembly generation)
      ↓
[C++] CUDA Runtime Execution (Stream dispatch, Pinned Memory pools)
```

## 2. Monorepo Folder Structure

The C++ backend codebase resides in the root `/backend` directory (and associated peer directories like `/mlir`, `/optimizer`).

```text
/backend
├── mlir/          # Core context management and MLIR DialectRegistries.
├── dialects/      # TableGen (.td) and C++ definitions for the Taria dialect.
├── lowering/      # Pass implementations (e.g., ASTToTaria, LinalgToGPU).
├── optimizer/     # The Tensor Optimization Engine (Fusion, Tiling, Graph rewrites).
├── scheduler/     # GPU Warp and Async Stream scheduling passes.
├── nvptx/         # LLVM Translation interfaces for PTX emission.
├── memory/        # Allocator lowering (Bufferization to MemRef).
├── compression/   # Lowering logic specific to Latent/VQ ops.
├── async/         # Lowering of `async pipeline` to CUDA event streams.
├── distributed/   # Lowering of `@sharded` tensors to NCCL/GPU communication ops.
├── profiling/     # Passes that inject PTX telemetry and clock-cycle counters.
└── api/           # C++ compiler invocation APIs (PassManager setup).
```

### Responsibilities & Ownership
- **`optimizer/`**: Owns the high-level semantic transformations. Operates *before* bufferization. Matches tensor graphs to apply kernel fusion.
- **`lowering/`**: Owns the mechanical translation between dialect abstractions. Strictly adheres to MLIR's Dialect Conversion framework.
- **`memory/`**: Owns the transition from value-semantics (Tensors) to memory-semantics (MemRefs). Implements bufferization, handling unified/pinned memory traits based on Taria's Semantic IR hints.

## 3. Future Scalability

The backend is aggressively isolated from the target hardware until the `GPU Dialect` phase.
To support future targets like AMD ROCm or Google TPUs:
- A new `rocdl/` or `xla/` directory is added parallel to `nvptx/`.
- The `GPU -> NVVM` pass is swapped out dynamically in the PassManager for a `GPU -> ROCDL` pass, natively generating AMDGCN assembly without touching the Taria semantic layer.
- `async pipeline` lowers to Vulkan Command Buffers instead of CUDA Streams.
