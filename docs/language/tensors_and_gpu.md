# Taria Tensors & GPU Execution System

Taria is built fundamentally around Tensors and GPU compute. Unlike Python frameworks that wrap C++ libraries, Taria compiles tensor operations directly into MLIR dialects and ultimately to PTX assembly.

## 1. Tensor System

Tensors in Taria are first-class, statically typed primitives. The type signature strictly encodes the data type, while shape and layout can be generic or concretely defined at compile-time.

### Tensor Declarations
```taria
# Statically shaped tensor (allocated at compile-time / register level)
chunk: Tensor<f32, [1024, 1024]> = Tensor.zeros()

# Dynamically shaped tensor (allocated in VRAM)
dynamic_chunk: Tensor<f32> = load_stream()
```

### Broadcasting and Slicing
Taria enforces NumPy-style broadcasting semantics but validates them at compile-time via MLIR shape inference where possible.

```taria
# Slicing
region = chunk[0:512, 512:1024]

# Broadcasting
bias: Tensor<f32, [1024]> = get_bias()
chunk += bias # Broadcasts bias across dimension 0
```

### Tensor Layouts and Memory Semantics
Taria allows explicit memory layout declarations to optimize cache hits on GPU hardware.

```taria
# Forces a Z-Curve tiled layout for spatial locality during convolutions
spatial_map: Tensor<f16, Layout=TiledZCurve> = ...
```

---

## 2. GPU Kernel System

Taria uses the `@gpu.kernel` decorator to elevate a standard function into a hardware-accelerated compute kernel.

### Thread and Block Mapping
The compiler automatically maps the outermost tensor dimensions to CUDA Grid and Block dimensions, but explicit control is available.

```taria
@gpu.kernel(
    block_size=256,
    shared_memory=true
)
fn spatial_reduce(input: Tensor<f32, [1024, 1024]>) -> Tensor<f32, [1024]> {
    # 'gpu.tile' pulls data from Global Memory (VRAM) to Shared Memory (SRAM)
    smem_tile: SharedTensor<f32, [256]> = gpu.tile(input, size=256)

    # Reductions over SharedTensors use warp-synchronous primitives
    return smem_tile.sum(dim=1)
}
```

### Compiler-Assisted Tensor Fusion
When the compiler encounters sequential element-wise or reduction operations, it fuses them into a single kernel launch to minimize VRAM bandwidth overhead.

```taria
# The compiler fuses this entire function into a single MLIR Linalg generic loop,
# executing the addition, multiplication, and sigmoid activation in one pass.
@gpu.kernel(block_size=512)
fn fused_activation(x: Tensor<f16>, w: Tensor<f16>, b: Tensor<f16>) -> Tensor<f16> {
    linear = (x * w) + b
    return linear.sigmoid()
}
```

---

## 3. Async Execution Model

Taria abandons the synchronous CPU-blocking model. GPU operations are implicitly asynchronous, utilizing CUDA streams under the hood.

### Async Pipelines
```taria
async pipeline compress_stream {
    # Stream reading happens concurrently with GPU computation
    chunk: Tensor<f16> = stream.read()

    # The compiler schedules this on Compute Stream A
    latent = encoder(chunk)

    # Scheduled on Compute Stream B when 'latent' is ready
    quantized = latent.quantize(bits=8)

    # Overlapped memory transfer back to host
    stream.write(quantized)
}
```

### Synchronization
Developers must explicitly await data if crossing the GPU/CPU boundary.
```taria
result: Tensor<f32> = async_kernel(data)
cpu_value: f32 = result[0, 0].await() # Blocks CPU until stream finishes
```
