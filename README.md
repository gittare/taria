# Taria: GPU Compression DSL Compiler

Taria is a next-generation, Python-superset Domain-Specific Language (DSL) and compiler for ultra-high-performance GPU computing, semantic compression, and AI-native tensor programming. It is designed for research and production in neural compression, autoencoder-based encoding, and distributed GPU execution.

---

## Features
- **Python-like syntax** with Rust-grade safety
- **MLIR-based IR** with custom Taria dialect
- **LLVM/NVPTX backend** for CUDA GPUs
- **Zero-cost abstractions** and high-performance codegen
- **Distributed GPU and async runtime**
- **Semantic compression primitives** (autoencoders, quantization, entropy models)
- **Extensible for future AI-native optimizations**

---

## Project Structure

```
taria/
├── frontend/      # Rust lexer, parser, AST, CLI
├── backend/       # C++/MLIR lowering, codegen
├── runtime/       # CUDA runtime, scheduler
├── dialects/      # MLIR dialects (Taria, etc.)
├── passes/        # MLIR passes (lowering, optimization)
├── ffi/           # Rust <-> C++ FFI bridge
├── include/       # Public C/C++ headers
├── tests/         # Unit, integration, E2E tests
├── examples/      # Example Taria programs
├── benchmarks/    # Performance benchmarks
├── tools/         # CLI tools, JIT, profilers
├── docs/          # Documentation, roadmap, engineering
└── .github/       # CI/CD, workflows
```

---

## Build & Test

### Rust Frontend
```
cargo build --workspace
cargo test --workspace
```

### C++/CUDA Backend
```
cmake .
make
ctest
```

---

## Compiler Pipeline
1. **Lex/Parse (Rust)** → AST
2. **AST → MLIR (FFI bridge)**
3. **MLIR Passes (C++/MLIR)**
4. **Lowering to GPU IR**
5. **NVPTX/PTX Codegen (LLVM)**
6. **CUDA Runtime Execution**

---

## Example: Taria Source
```python
@gpu.kernel(block_size=256)
def compress_chunk(chunk: Tensor) -> CompressedChunk:
    latent = encoder(chunk)
    return quantize(latent)
```

### Corresponding MLIR
```mlir
taria.gpu_kernel @compress_chunk(%chunk: tensor<f32>) -> tensor<i8> {
  %latent = taria.encode %chunk : tensor<f32> -> tensor<f32>
  %quant = taria.quantize %latent : tensor<f32> -> tensor<i8>
  taria.return %quant : tensor<i8>
}
```

---

## Key Engineering Practices
- **SSA construction** and IR immutability
- **Arena allocation** for AST/IR nodes
- **Borrow-safe Rust and smart-pointer C++**
- **GPU register pressure and warp divergence minimization**
- **Tensor memory alignment and async pipeline scheduling**
- **Profile-guided and auto-tuning optimization**
- **JIT/AOT hybrid kernel compilation and caching**

---

## Roadmap & Futures
- ROCm, Vulkan/SPIR-V, TPU, Metal backends
- Distributed/federated execution, tensor sharding
- AI-native compiler optimization, neural cost models
- Plugin system, LSP/IDE, Jupyter integration
- Secure, reproducible, and auditable builds

See `docs/roadmap.md` and `docs/futures.md` for details.

---

## Contributing
- Follow LLVM/MLIR and Rust best practices
- Write tests for all new features
- Document all public APIs
- Ensure ABI stability in FFI
- See `docs/compiler_engineering.md` for advanced guidance

---

## License
Taria is released under the Apache 2.0 License.

---

## Contact & Community
- [GitHub Issues](https://github.com/your-org/taria/issues)
- [Discussions](https://github.com/your-org/taria/discussions)
- [Contributing Guide](docs/CONTRIBUTING.md)

---

Taria: The AI-native, GPU-first compiler for semantic compression and next-generation tensor programming.
