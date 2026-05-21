# Backend Lowering and Optimization Engines

The translation from MLIR's `taria` dialect to bare-metal PTX relies on aggressive, specialized optimization passes.

## 1. The Tensor Optimization Engine (`optimizer/`)

Before translating semantic constructs into loops, Taria applies graph-level optimizations.

### Tensor Fusion
The `TensorFusionOptimizer` uses MLIR's `GreedyPatternRewriteDriver` to locate sequential element-wise, reduction, or compression operations.
- **Rewrite Rule**: `taria.encode` followed by `taria.compress.quantize` is wrapped in a `taria.tensor.fuse` block.
- **Heuristics**: Fusion stops if register pressure estimates exceed 255 registers per thread (which would crash occupancy down to 0), or if the intermediate tensor must be fanned out to multiple non-fusable consumers.

### Tensor Tiling
The `LayoutOptimizer` pass inspects `!taria.tensor` layout metadata.
- If `TiledZCurve` is requested, the access indices inside the resulting `linalg.generic` are transformed using Affine Maps to calculate Morton codes, vastly improving L1 cache hit rates for spatial convolutions.

## 2. Memory Optimization System (`memory/`)

When moving from `taria` (value semantics) to `linalg` (memory semantics), bufferization occurs.

### Memory Pools and Pinned Memory
If a tensor originates from an `async pipeline` streaming read, the bufferization pass assigns it a `Pinned` memory trait. The lowering to the CUDA Runtime API emits calls to the `MemoryPool::allocate_pinned()` rather than a standard `cudaMalloc`.

### Memory Coalescing & Shared Memory (SRAM)
The `OccupancyOptimizer` pass analyzes thread access patterns in `linalg` loops.
- **Coalescing**: If innermost loop strides are not multiples of 32 (warp size), padding ops are inserted. This guarantees that `llvm.load` emits 128-byte `ld.global` PTX instructions.
- **SRAM Staging**: High-reuse tensors (like quantization codebooks) are identified. The pass inserts `gpu.subgroup_mma` operations to orchestrate cooperative warp-loading of the tensor from VRAM to SRAM (`__shared__`) before the main compute loop begins.

## 3. GPU Scheduler (`scheduler/`)

The `AsyncScheduler` pass analyzes independent dataflow graphs.
- Independent `taria.gpu.launch` nodes are assigned to different CUDA Streams via `async.execute`.
- **Async Overlap**: The scheduler ensures memory transfer nodes (`gpu.memcpy`) are placed on copy-engine streams, while math kernels are on compute streams, allowing simultaneous execution on the GPU hardware.

## 4. PTX Optimization (`nvptx/`)

The final translation from `NVVM Dialect` to LLVM IR enables target-specific hardware optimization.

- **Register Optimization**: Local variables in `llvm.func` are aggressively promoted to registers (`%f1`, `%r1`).
- **Warp-Level Shuffles**: Reductions and nearest-neighbor searches are lowered using LLVM NVPTX intrinsics mapped to `__shfl_sync` (PTX `shfl.sync.bfly`), allowing threads within a warp to exchange data without writing to Shared Memory.
- **Instruction Level Parallelism (ILP)**: By unrolling inner loops and fusing multiply-add operations into `fma.rn.f32` instructions, the PTX generator minimizes clock cycles per instruction.
