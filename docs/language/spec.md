# Taria Core Language Specification

Taria is a statically typed, compiled, GPU-first Domain-Specific Language (DSL) engineered for AI systems programming, tensor manipulation, and semantic compression. It marries Pythonic ergonomics with Rust's memory safety and CUDA's execution control.

## 1. Syntax and Core Paradigms

Taria uses significant whitespace (indentation) for block scoping, similar to Python, but enforces explicit static typing and ownership constraints.

### Variables and Immutability
Variables are immutable by default. The `mut` keyword is required for mutable bindings.
```taria
# Immutable binding
val: f32 = 3.14

# Mutable binding
mut counter: i32 = 0
counter += 1
```

### Functions and Return Types
Functions are defined using the `fn` keyword. Types are mandatory for parameters and return values.
```taria
fn compute_entropy(prob: Tensor<f32>) -> Tensor<f32> {
    # -p * log2(p)
    return -prob * prob.log2()
}
```

## 2. The Type System & Ownership

Taria's type system is built around strict data ownership to ensure memory safety without a garbage collector—critical for managing gigabytes of VRAM.

### Primitive Types
- `i8`, `i16`, `i32`, `i64`
- `u8`, `u16`, `u32`, `u64`
- `f16`, `bf16`, `f32`, `f64`
- `bool`

### Ownership and References
When managing large tensors, copying data implicitly is catastrophic for performance. Taria enforces affine types (move semantics).

```taria
fn process(data: Tensor<f32>) -> Tensor<f32> { ... }

fn main() {
    chunk: Tensor<f32> = load("data.bin")

    # Ownership of 'chunk' is moved to process().
    result = process(chunk)

    # Error: 'chunk' is no longer valid here.
    # print(chunk.shape)
}
```

To borrow memory without taking ownership, Taria uses read-only `&` and mutable `&mut` references.
```taria
fn normalize(data: &mut Tensor<f32>) {
    data /= data.max()
}
```

### Structs, Enums, and Pattern Matching
Taria supports algebraic data types and exhaustive pattern matching.

```taria
enum CodecMode {
    Lossless,
    Lossy(u8) # Stores the quantization bit-depth
}

fn configure_codec(mode: CodecMode) {
    match mode {
        CodecMode.Lossless => setup_ans(),
        CodecMode.Lossy(bits) => setup_vq(bits),
    }
}

struct CompressionConfig {
    block_size: i32,
    mode: CodecMode,
}
```

## 3. Module System and Visibility

Taria uses a clear, hierarchical module system. By default, items are private. The `pub` keyword exposes them.

```taria
# file: src/math/entropy.taria

pub fn shannon_entropy(data: &Tensor<f32>) -> f32 {
    ...
}

# file: src/main.taria
import math.entropy

fn main() {
    val = entropy.shannon_entropy(&my_tensor)
}
```

### Generics and Traits
Taria uses traits (interfaces) to bound generic types.

```taria
trait Quantizable {
    fn quantize(self, bits: u8) -> Tensor<i8>
}

impl Quantizable for LatentVector<f32> {
    fn quantize(self, bits: u8) -> Tensor<i8> {
        return gpu.vq.nearest_neighbor(self, bits)
    }
}

fn encode_any<T: Quantizable>(data: T) -> Tensor<i8> {
    return data.quantize(bits=8)
}
```
