# Tensor Fusion Optimization Example

## Purpose
This example demonstrates the Taria compiler's ability to fuse adjacent, memory-bound tensor operations into a single compute-bound kernel using MLIR greedy pattern rewrites.

## Target Audience
- Compiler Optimization Engineers
- High-Performance GPU Programmers

## MLIR Transformation Pipeline
In standard Python/PyTorch, adding a scalar and then multiplying by a scalar results in two kernel launches and three VRAM round trips. Taria identifies these element-wise operations as `linalg.generic` loops during MLIR lowering and fuses the loop bodies. The resulting PTX has one load, fused math instructions (FMA), and one store.
