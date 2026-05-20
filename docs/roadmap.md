# Taria Compiler Future Roadmap

## Near-Term
- ROCm backend: Add MLIR lowering to ROCm dialect, support AMD GPUs.
- Vulkan/SPIR-V backend: Lower to MLIR SPIR-V dialect, target Vulkan compute.
- TPU backend: Integrate with XLA/HLO, emit TPU ops.
- Distributed training: Add tensor sharding, NCCL/collective ops.
- Tensor sharding: Partition tensors across devices, manage halo exchange.
- AI graph optimization: Fuse ops, reorder for memory locality, auto-tune layouts.
- Automatic mixed precision: Insert cast ops, propagate FP16/FP32 as needed.

## Mid-Term
- Speculative kernel fusion: Profile at runtime, fuse hot kernels.
- Neural compiler optimization: Use learned cost models for scheduling, fusion, tiling.
- Auto-tuning passes: Integrate with TVM-style auto-tuner for block/grid size, tiling, fusion.
- JIT/AOT hybrid: Support both dynamic and static kernel compilation.

## Long-Term
- End-to-end distributed GPU compilation and execution.
- Full AI-native optimization pipeline.
- Integration with major ML frameworks (PyTorch, TensorFlow, JAX).
- Community-driven dialect/plugin ecosystem.
