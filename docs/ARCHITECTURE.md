# Taria Architecture Guide

## Compiler Pipeline Flow
1. **Frontend (Rust)**: Parses Pythonic DSL into an AST. Employs zero-copy lexing and arena allocation for performance.
2. **FFI Bridge (C API)**: Hands off the immutable, source-mapped AST representation to the MLIR/C++ backend.
3. **MLIR Layer (C++)**:
   - Lowers AST to `Taria` MLIR dialect.
   - Lowers `Taria` to `Linalg` or Standard GPU dialects.
4. **Backend/Codegen**: Lowers GPU dialects to NVVM, then PTX.
5. **Runtime (CUDA)**: Manages stream execution, asynchronous scheduling, and memory pooling.

## Key Design Principles
- **Borrow-Safe AST Construction**: The Rust frontend produces immutable AST trees to easily share across threads without locking.
- **SSA in MLIR**: All intermediate operations in MLIR strictly adhere to Static Single Assignment to simplify dataflow analysis and optimization.
- **Zero-Cost Abstractions**: The FFI boundary defines an ownership model where C++ acquires ownership of AST nodes temporarily during translation.
- **Semantic Compression**: Native operations (`taria.encode`, `taria.decode`, `taria.compress`, `taria.quantize`) allow domain-specific optimizations (like fusing quantization with encoding).

## Example End-to-End
```python
@gpu.kernel(block_size=256)
def compress_chunk(chunk: Tensor) -> CompressedChunk:
    latent = encoder(chunk)
    return quantize(latent)
```
1. Frontend creates AST `FunctionDecl` with decorators.
2. FFI exports opaque handle.
3. C++ builds MLIR `taria.gpu_kernel` containing `taria.encode` and `taria.quantize`.
4. MLIR lowers to NVVM operations.
5. CUDA Runtime schedules the resulting PTX binary.
