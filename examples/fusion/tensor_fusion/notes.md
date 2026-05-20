# Optimization Notes

## Memory Coalescing
The tensor access pattern `affine_map<(d0) -> (d0)>` translates to a linear, stride-1 mapping on the GPU. When 32 threads in a warp request `[0..31]`, the hardware coalesces this into a single 128-byte memory transaction.

## Fused Multiply Add (FMA)
The LLVM NVPTX backend is highly intelligent. By fusing the AST elements in MLIR, the final lowering step automatically recognizes a multiplication followed immediately by an addition, and emits the `fma.rn.f32` instruction. This executes both operations in a single clock cycle on the SM, halving the instruction latency.

## VRAM Bandwidth
This is the true bottleneck for element-wise operations. A naive compiler allocates an intermediate buffer for the output of the multiplication. This requires writing 4MB to VRAM and immediately reading 4MB back. Taria's graph fusion eliminates this, moving the data exclusively through the register file (`%f1`, `%f2`).
