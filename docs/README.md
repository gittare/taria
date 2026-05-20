# Taria Compiler Documentation

## Overview
Taria is a Python-superset DSL and compiler for GPU-native semantic compression and tensor programming. It features a Rust frontend, MLIR-based IR, C++/CUDA backend, and a high-performance GPU runtime.

## Directory Structure
- `frontend/`: Rust lexer, parser, AST, CLI
- `backend/`: C++/MLIR lowering, codegen
- `runtime/`: CUDA runtime, scheduler
- `dialects/`: MLIR dialects (Taria, etc.)
- `passes/`: MLIR passes (lowering, optimization)
- `ffi/`: Rust <-> C++ FFI bridge
- `include/`: Public C/C++ headers
- `tests/`: Unit, integration, E2E tests
- `examples/`: Example Taria programs
- `benchmarks/`: Performance benchmarks
- `tools/`: CLI tools, JIT, profilers
- `docs/`: Documentation

## Build Instructions
- Rust: `cargo build --workspace`
- C++/CUDA: `cmake . && make`
- Tests: `cargo test --workspace`, `ctest`

## Compiler Pipeline
1. Lex/Parse (Rust) → AST
2. AST → MLIR (FFI)
3. MLIR Passes (C++/MLIR)
4. Lowering to GPU IR
5. NVPTX/PTX Codegen (LLVM)
6. CUDA Runtime Execution

## Contribution Guide
- Follow LLVM/MLIR and Rust best practices
- Write tests for all new features
- Document all public APIs
- Use arena allocation for AST/IR
- Ensure ABI stability in FFI

## Roadmap
- ROCm, Vulkan, TPU backends
- Distributed GPU compilation
- Neural compiler optimization
- Auto-tuning and mixed precision
