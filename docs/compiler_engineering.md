# Advanced Compiler Engineering Guidance

## SSA Construction
- Use MLIR's SSA form for all IR. Ensure unique value names, dominance, and phi nodes for control flow.
- In Rust, keep AST/IR immutable after construction to simplify analysis and transformation.

## Arena Allocation
- Use `bumpalo` or `typed-arena` in Rust for AST/IR nodes. In C++, use `llvm::BumpPtrAllocator` for MLIR/LLVM IR.
- Avoid per-node heap allocations for performance and cache locality.

## Borrow-safe Compiler Design
- Leverage Rust's lifetimes and ownership for safe AST/IR traversal. Avoid `Rc` cycles; prefer explicit arenas.
- In C++, use smart pointers and clear ownership boundaries for IR objects.

## GPU Register Pressure Reduction
- Minimize live ranges in generated code. Use local memory for spills. Prefer scalarization of tensor ops when possible.

## Warp Divergence Minimization
- Structure control flow to minimize divergent branches. Use predication and warp-synchronous programming where possible.

## Tensor Memory Alignment
- Align all tensor allocations to 128B for optimal memory transactions. Use `__align__` in CUDA and MLIR layout attributes.

## Async Pipeline Scheduling
- Overlap data transfers and kernel launches using CUDA streams and events. Use command queues and dependency graphs.

## Profile-Guided Optimization
- Instrument kernels, collect runtime stats, and feed back into kernel fusion/block size selection.

## JIT Compilation
- Use MLIR ExecutionEngine or LLVM ORC JIT for dynamic kernel generation. Cache JITed kernels by hash of source/params.

## AOT Compilation
- Precompile kernels for target architectures. Cache PTX/cubin artifacts per device/arch.

## Kernel Caching
- Hash kernel source/params, cache binaries per device/arch. Use persistent storage for large-scale deployments.

## Distributed GPU Compilation
- Use gRPC/ZeroMQ for distributed build. Synchronize kernel caches and IR artifacts across nodes.
