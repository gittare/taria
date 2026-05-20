// 1. Initial lowering from AST to Taria Dialect
module {
  taria.gpu_kernel block_size(256) shared_mem_bytes(65536) {
    %latent = taria.encode %chunk : tensor<1024x1024xf32> -> tensor<32x32xf32>
    %quant = taria.vector_quantize %latent, %codebook : tensor<32x32xf32>, tensor<256x64xf32> -> tensor<32x32xi8>
    taria.return %quant : tensor<32x32xi8>
  }
}

// 2. After Fusion and Linalg Lowering
module {
  gpu.module @compress_kernel {
    gpu.func @compress_chunk(%arg0: memref<1024x1024xf32>, %arg1: memref<256x64xf32>, %out: memref<32x32xi8>) workgroup_size = [256, 1, 1] {
      // Allocate shared memory for the codebook
      %codebook_smem = memref.alloc() : memref<256x64xf32, 3>
      gpu.barrier
      // ... loops omitted ...
      // Compute latent features in registers
      // ... Linalg Matmul / Conv ...
      // Quantize using %codebook_smem
      // Store to %out
      gpu.return
    }
  }
}
