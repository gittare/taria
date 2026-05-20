# Taria Compiler Documentation Ecosystem

Welcome to the official documentation for **Taria**, a GPU-native Domain-Specific Language (DSL) and compiler infrastructure engineered for extreme-scale semantic tensor compression, latent-space encoding, and ultra-high-performance GPU execution.

This documentation system is designed to onboard and support elite compiler engineers, MLIR/LLVM contributors, GPU systems researchers, and ML infrastructure engineers. It maintains the rigorous technical standards expected of projects like LLVM, MLIR, Triton, and PyTorch Internals.

---

## Directory Structure & Guide

### 1. `architecture/`
- **Purpose:** Details the global system architecture, bridging the Rust frontend, FFI, MLIR, LLVM, and CUDA runtime.
- **Audience:** System architects, core contributors, and researchers.
- **Included Files:** `pipeline.md`, `data_flow.md`, `ownership.md`
- **Engineering Focus:** Explains ownership boundaries, the exact MLIR lowering flow (AST -> Taria -> Linalg -> GPU -> NVVM -> PTX), and runtime scheduling diagrams.
- **Future Expansion:** Distributed execution roadmaps and multi-GPU memory topologies.

### 2. `compiler/`
- **Purpose:** Documents the internal phases of the Taria compiler.
- **Audience:** Compiler engineers.
- **Included Files:** `phases.md`, `ssa_generation.md`, `caching.md`
- **Engineering Focus:** Focuses on SSA generation, incremental compilation strategies, pass pipelines, and AOT vs JIT compilation tradeoffs.

### 3. `frontend/`
- **Purpose:** Details the Rust-based frontend infrastructure.
- **Audience:** Language designers and Rust frontend engineers.
- **Included Files:** `lexer_design.md`, `pratt_parser.md`, `ast_arena.md`, `diagnostics.md`
- **Engineering Focus:** Zero-copy SIMD-aware tokenization, Pratt parsing for expressions, immutable arena-allocated AST generation, and `rustc`-style colored diagnostics and source mapping.

### 4. `backend/`
- **Purpose:** Documents the C++ LLVM/NVVM backend lowering pipeline.
- **Audience:** LLVM engineers, code generation specialists.
- **Included Files:** `llvm_integration.md`, `ptx_generation.md`, `register_allocation.md`
- **Engineering Focus:** The specific translations from the Standard GPU dialect into NVVM and LLVM IR, PTX code emission, and register pressure optimization strategies.

### 5. `mlir/`
- **Purpose:** Extensive documentation on the custom `taria` MLIR dialect.
- **Audience:** MLIR infrastructure engineers.
- **Included Files:** `taria_dialect.md`, `ods_definitions.md`, `rewrite_patterns.md`
- **Engineering Focus:** ODS (Operation Definition Specification) for ops like `taria.encode` and `taria.quantize`, type inference, canonicalization hooks, and dialect conversion to Linalg.

### 6. `runtime/`
- **Purpose:** Details the C++/CUDA execution environment.
- **Audience:** Runtime engineers and systems programmers.
- **Included Files:** `async_execution.md`, `memory_pools.md`, `scheduler.md`
- **Engineering Focus:** Asynchronous CUDA stream management, unified/pinned memory pooling for zero-copy PCIe transfers, and the PTX closure scheduler.

### 7. `gpu/`
- **Purpose:** Deep dive into GPU-specific performance engineering.
- **Audience:** CUDA optimization engineers and GPU architects.
- **Included Files:** `execution_model.md`, `memory_coalescing.md`, `occupancy.md`
- **Engineering Focus:** Warp execution semantics, shared memory (SRAM) utilization, tensor tiling, kernel fusion, and mitigating thread divergence in latent search spaces.

### 8. `ffi/`
- **Purpose:** Documents the Rust ↔ C++ bridge.
- **Audience:** Systems programmers managing language interoperability.
- **Included Files:** `c_abi.md`, `ownership_lifecycle.md`
- **Engineering Focus:** ABI stability, opaque handle transfers, `catch_unwind` panic isolation, and memory lifecycle management across the boundary.

### 9. `optimization/`
- **Purpose:** Strategies for compile-time and runtime tensor optimizations.
- **Audience:** Compiler optimization engineers.
- **Included Files:** `tensor_fusion.md`, `greedy_rewrites.md`, `layout_optimization.md`
- **Engineering Focus:** MLIR greedy pattern rewrite systems, fusing autoencoders with quantization, and memory layout transformations for contiguous warp access.

### 10. `compression/`
- **Purpose:** Research-grade documentation on semantic compression theory.
- **Audience:** ML researchers and AI infrastructure engineers.
- **Included Files:** `semantic_compression.md`, `latent_encoding.md`, `entropy_modeling.md`
- **Engineering Focus:** Theoretical and applied tensor representation in latent space, vector quantization codebook lookups, and AI-native neural codecs.

### 11. `language/`
- **Purpose:** The language specification for Taria DSL.
- **Audience:** Developers writing Taria code.
- **Included Files:** `syntax.md`, `decorators.md`, `tensor_types.md`
- **Engineering Focus:** Python-superset syntax rules, `@gpu.kernel` memory semantics, strong typing for tensor dimensions, and async primitives.

### 12. `tutorials/`
- **Purpose:** Step-by-step guides for learning Taria.
- **Audience:** Beginners and new contributors.
- **Included Files:** `getting_started.md`, `mlir_lowering_tutorial.md`, `custom_pass.md`
- **Engineering Focus:** Realistic build instructions, debugging workflows, and writing your first MLIR rewrite pass in C++.

### 13. `benchmarks/`
- **Purpose:** Methodology for performance tracking.
- **Audience:** Performance engineers.
- **Included Files:** `methodology.md`, `nvprof_workflows.md`
- **Engineering Focus:** GPU profiling (Nsight/nvprof), measuring compression throughput (GB/s), and verifying occupancy and memory bandwidth utilization.

### 14. `contributing/`
- **Purpose:** Workflow and coding standards for the project.
- **Audience:** Open-source contributors.
- **Included Files:** `rust_style_guide.md`, `llvm_coding_standards.md`, `rfc_process.md`
- **Engineering Focus:** CI/CD integration, FileCheck testing strategies for MLIR passes, and the Pull Request/RFC workflow.

### 15. `internals/`
- **Purpose:** Advanced deep-dives into compiler memory and state.
- **Audience:** Core maintainers.
- **Included Files:** `pass_manager.md`, `arena_allocators.md`
- **Engineering Focus:** Understanding the global context, thread-local storage in the parser, and runtime tensor allocators.

### 16. `api/`
- **Purpose:** Public API references.
- **Audience:** Embedders and integration engineers.
- **Included Files:** `rust_api.md`, `cpp_api.md`
- **Engineering Focus:** Embedding the Taria compiler in PyTorch, C++ API examples, and runtime linkage.

### 17. `roadmap/`
- **Purpose:** Short-term and long-term project planning.
- **Audience:** Investors, researchers, and core team.
- **Included Files:** `v1_roadmap.md`, `backend_expansion.md`
- **Engineering Focus:** Plans for AMD ROCm, Vulkan/SPIR-V, TPU (XLA HLO) generation, and distributed NCCL tensor sharding.

### 18. `research/`
- **Purpose:** Whitepapers and literature reviews.
- **Audience:** Academic and industry researchers.
- **Included Files:** `neural_encoding_papers.md`, `distributed_scheduling.md`
- **Engineering Focus:** The intersection of compiler theory and neural data reduction.
