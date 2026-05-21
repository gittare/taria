# Taria Standard Library (stdlib)

Taria ships with a zero-dependency, GPU-native standard library designed to abstract away raw PTX manipulation while maintaining maximum performance.

## `taria.tensor`

The core math and tensor manipulation library. All operations here are lowered to `linalg` generics and fused when chained.

```taria
import taria.tensor

fn math_example() {
    a: Tensor<f32, [100]> = tensor.ones()
    b: Tensor<f32, [100]> = tensor.randn(mean=0.0, std=1.0)

    # Element-wise operations (Automatically Fused)
    c = (a + b).exp()

    # Reductions
    max_val: f32 = c.max()
}
```

## `taria.gpu`

Provides low-level access to the execution hierarchy, allowing elite programmers to optimize occupancy and memory flow.

```taria
import taria.gpu

@gpu.kernel(block_size=128)
fn custom_reduction(input: Tensor<f32>) -> Tensor<f32> {
    # Intrinsic: Sync threads within a warp (32 threads)
    gpu.sync_warp()

    # Intrinsic: Barrier across the entire block (128 threads)
    gpu.sync_block()

    # Intrinsic: Retrieve thread ID
    tid: u32 = gpu.thread_id().x

    # Allocating explicit Shared Memory
    smem: SharedMemory<f32, [128]> = gpu.alloc_shared()

    # ... logic ...
    return output
}
```

## `taria.compress`

The flagship module containing AI-native compression primitives.

```taria
import taria.compress
import taria.compress.models

fn encode_data(data: Tensor<f16>) -> Bitstream {
    # Pre-compiled MLIR models stored in Taria Model Format (.tmf)
    encoder = models.load_autoencoder("resnet_v1.tmf")

    latent: LatentVector<f16> = encoder(data)

    # Hardware-accelerated nearest-neighbor search
    quantized = compress.vq.nearest_neighbor(latent, codebook_size=4096)

    # Hardware-accelerated Asymmetric Numeral Systems encoding
    return compress.entropy.encode(quantized)
}
```

## `taria.distributed` (Roadmap)

Primitives for multi-GPU execution using NCCL rings under the hood.

```taria
import taria.distributed as dist

fn multi_gpu_compress(data: Tensor<f32>) {
    # Shards the tensor across 8 GPUs automatically
    sharded_data = dist.shard(data, devices=8)

    # Executes the pipeline on all 8 GPUs independently
    dist.parallel_execute(compress_pipeline, sharded_data)
}
```
