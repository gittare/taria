// 1. Initial Taria MLIR Dialect (Unfused)
// ...
// %t1 = taria.mul %input, %scale : tensor<1000000xf32>
// %out = taria.add %t1, %shift : tensor<1000000xf32>
// ...

// 2. Post-Fusion Linalg Dialect
// The LinalgFusion pass has merged the two element-wise operations into one loop body.
module {
  func.func @scale_and_shift(%arg0: memref<1000000xf32>, %scale: f32, %shift: f32, %out: memref<1000000xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%arg0 : memref<1000000xf32>) outs(%out : memref<1000000xf32>) {
    ^bb0(%in: f32, %out_val: f32):
      // Fused FMA (Fused Multiply Add) calculation
      %0 = arith.mulf %in, %scale : f32
      %1 = arith.addf %0, %shift : f32
      linalg.yield %1 : f32
    }
    return
  }
}
