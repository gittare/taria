# Taria Compiler Examples Ecosystem

Welcome to the Taria Examples Ecosystem. This directory serves as a comprehensive, elite-level curriculum designed to teach the Taria language, demonstrate MLIR lowering mechanics, and showcase ultra-high-performance GPU execution strategies for semantic compression.

## Intended Audience
- **Compiler Engineers**: Learn how Taria AST lowers through MLIR dialects down to LLVM and PTX.
- **GPU Researchers & Systems Engineers**: Explore explicit memory coalescing, shared memory allocations, occupancy limits, and async CUDA stream scheduling.
- **ML & AI Infrastructure Engineers**: Master semantic autoencoder pipelines, vector quantization, and latent-space tensor manipulation.

## Directory Structure

Each directory contains fully functional Taria scripts accompanied by expected MLIR lowering output, generated PTX assembly, CMake build instructions, and Python benchmarking scripts.

- **`basics/`**: Syntax, decorators, simple tensor declarations, and introductory `@gpu.kernel` wrappers.
- **`tensors/`**: Physical memory layouts, dimension shuffling, and broadcasting semantics.
- **`gpu/`**: Explicit thread/block execution mappings, warp synchronization, and occupancy optimization strategies.
- **`kernels/`**: Deep dives into tiled GEMMs, convolutions, and memory-coalesced reductions.
- **`compression/`**: Core semantic compression pipelines combining neural encoders and entropy modeling.
- **`autoencoders/`**: MLIR-driven semantic feature extraction and latent projections.
- **`latent/`**: Theory and implementation of latent vector interpolation and transformation.
- **`quantization/`**: Vector Quantization (VQ), codebook mapping, and low-bit entropy coding.
- **`runtime/`**: Integration with the `CudaScheduler` and `MemoryPool` (pinned zero-copy memory).
- **`async/`**: Overlapping compute with memory transfers via multi-stream pipelines.
- **`optimization/`**: Compiler rewrites enforcing cache locality and register pressure reduction.
- **`fusion/`**: Demonstrations of MLIR `RewritePattern`s folding graph sequences into single-launch kernels.
- **`distributed/`**: Tensor sharding and NCCL ring execution semantics for multi-GPU scaling.
- **`mlir/`**: Pure compiler-engineering examples showing pass-by-pass dialect conversions.
- **`ptx/`**: Annotated LLVM IR and resulting NVIDIA PTX assembly for hardware-level verification.
- **`benchmarking/`**: Nsight Compute and `nvprof` workflows for profiling throughput and occupancy.
- **`interoperability/`**: FFI integration, bridging Taria kernels with Rust and C++ host processes.
- **`research/`**: Cutting-edge AI-guided scheduling, learned memory layouts, and neural codecs.
- **`advanced/`**: End-to-end, multi-stage streaming semantic compression systems for production deployment.

## How to Build and Run Examples

Each example directory contains a `CMakeLists.txt` for compiling the C++ runtime bindings, a `main.taria` source file, and a `benchmark.py` script.

Example execution:
```bash
cd examples/compression/semantic_chunk
cmake -B build
make -C build
python benchmark.py --visualize-occupancy
```
