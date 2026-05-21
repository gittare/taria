# Taria Compiler Architecture

Taria utilizes a hybrid compiler stack leveraging Rust for frontend safety and ergonomics, and C++/MLIR/LLVM for world-class backend optimization and code generation.

## The Compilation Pipeline

The pipeline is strictly phased to ensure separation of concerns, utilizing an immutable intermediate state and Strict Single Assignment (SSA) form throughout the backend.

```text
Taria Source Code (.taria)
         ↓
    [ Rust Frontend ]
         ↓
      Lexer (Zero-copy, SIMD-aware)
         ↓
      Parser (Pratt + Recursive Descent)
         ↓
      AST (Arena-allocated, Immutable)
         ↓
      Semantic Analysis (Type Inference, Symbol Resolution)
         ↓
    [ FFI Bridge (C ABI) ]
         ↓
    [ C++ Backend (MLIR) ]
         ↓
      Taria MLIR Dialect (High-level semantic operations)
         ↓
      Linalg Dialect (Generic tensor loops and compute)
         ↓
      GPU Dialect (Thread/Block mapping, Shared Memory)
         ↓
      NVVM Dialect / AMDGCN (Target-specific lowering)
         ↓
      LLVM IR
         ↓
    [ LLVM Target Backend ]
         ↓
      PTX / ISA Assembly
         ↓
    [ Taria Runtime (CUDA/C++) ]
         ↓
      Asynchronous Stream Execution
```

## Frontend/Backend Separation

The boundary between the Rust frontend and the C++ MLIR backend is intentionally stark.
1. **Ownership Boundary**: The Rust frontend creates an AST. It transfers a strictly read-only, opaque handle (`TariaAstHandle`) across the FFI.
2. **Panic Isolation**: All Rust FFI boundaries are wrapped in `catch_unwind`. If the frontend fails, it returns a C-compatible error code rather than unwinding into C++ frames, which results in Undefined Behavior.
3. **MLIR Ingestion**: The C++ pass `ASTToTaria` traverses the opaque handle using C-API callbacks to instantiate MLIR `Operation`s within an `mlir::OpBuilder`.

## Semantic Compression Pipeline

Taria treats operations like vector quantization and autoencoding as primitive language constructs.

When a user writes:
```python
latent = encoder(input)
return quantize(latent)
```

The MLIR pipeline generates:
1. `taria.encode`
2. `taria.quantize`

By keeping these high-level, the `TariaOptimization` pass can recognize this specific pattern and apply **Kernel Fusion**. Instead of writing `latent` to global memory, the generated PTX will compute the latent vector in registers, perform the nearest-neighbor quantization against a codebook in Shared Memory (`__shared__`), and only write the `i8` quantized result to VRAM.

## Distributed Execution Roadmap

Future architecture extensions involve integrating NCCL primitives natively into the `Taria` MLIR dialect. Operations producing tensors annotated with `@sharded` will automatically emit `gpu.all_reduce` or `gpu.all_gather` MLIR nodes, allowing semantic compression to operate synchronously across multi-GPU rings without Python-level overhead.
