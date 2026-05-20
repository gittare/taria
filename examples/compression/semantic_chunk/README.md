# Semantic Chunk Compression Example

## Purpose
This example demonstrates Taria's core value proposition: fusing a neural latent encoder with vector quantization directly into a highly optimized, single-launch GPU kernel.

## Target Audience
- AI Infrastructure Engineers building neural codecs.
- Compiler Engineers studying MLIR kernel fusion.

## Compiler & GPU Concepts
Instead of executing an encoder block, saving it to global memory (VRAM), and then executing a quantizer block, Taria's `TariaOptimization` MLIR pass recognizes the sequence. It generates a kernel that calculates the latent space in registers, performs codebook lookup using Shared Memory (SRAM), and only writes the quantized `i8` integers to VRAM.

- **Occupancy:** Pinned to `block_size=256` for optimal SM scheduling.
- **Shared Memory:** The quantization codebook is explicitly loaded into SRAM to prevent VRAM latency.
