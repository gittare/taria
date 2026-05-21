# Notes on Performance and Optimization

## Warp Divergence
The nearest-neighbor search inside the `VectorQuantizer` is notoriously prone to warp divergence if implemented natively as a standard loops/branching structure. By pushing this down into the `taria` MLIR dialect, the compiler safely lowers this into `__shfl_sync` reduction trees. All 32 threads in a warp stay perfectly synchronized during the search phase.

## Memory Coalescing
The latent vector resulting from the `NeuralEncoder` has a trailing dimension of 64 `f32`s (256 bytes). This perfectly maps to 2 128-byte transactions for the memory controller, ensuring zero wasted bandwidth when saving the intermediate results back to registers or L1 cache.

## Kernel Fusion Tradeoffs
Fusing these kernels saves an immense amount of global memory writes (VRAM). However, it drastically increases register pressure per thread. We enforce `block_size=256` to ensure that we do not exceed the 64K register file limit per Streaming Multiprocessor (SM) on NVIDIA Ampere, allowing us to maintain 100% theoretical occupancy.
