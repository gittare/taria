# Semantic Compression Theory & Taria Implementation

Traditional lossless compression algorithms (LZ4, Zstd) look for exact byte repetitions. When applied to high-entropy floating-point tensor data (such as raw sensor data, video streams, or neural network activations), they fail to achieve meaningful compression ratios.

Taria introduces **Semantic Compression** natively at the compiler level. Instead of searching for exact byte patterns, semantic compression learns the underlying distribution of the data and represents it in a highly compressed latent space.

## The Semantic Compression Pipeline

Taria natively models the three stages of neural compression:

1. **Latent Space Encoding (`taria.encode`)**
2. **Vector Quantization (`taria.vector_quantize`)**
3. **Entropy Modeling (`taria.entropy_model`)**

### 1. Latent Space Encoding

Using a trained Neural Autoencoder, raw input tensors are projected into a lower-dimensional continuous latent space.
- *Example*: A `[1024, 1024, 3]` high-resolution tensor is reduced to a `[32, 32, 64]` feature map. The spatial redundancy has been removed by the neural network.

### 2. Vector Quantization (VQ)

To compress the continuous latent vectors into a bitstream, they must be discretized.
- Taria uses a learned codebook (a matrix of $K$ vectors).
- The `vector_quantize` operation compares every vector in the latent tensor against the codebook, replacing the continuous vector with the integer index of the closest codebook entry.
- *Example*: The `[32, 32, 64]` tensor of `f32` (256KB) becomes a `[32, 32]` tensor of `i16` indices (2KB).

### 3. Entropy Modeling

The integer indices from the VQ stage do not appear with equal probability. An entropy model predicts the likelihood of each index.
- Utilizing Asymmetric Numeral Systems (ANS) or Arithmetic Coding, the sequence of indices is compressed down to the Shannon entropy limit.
- Taria compiles these entropy encoders directly into hardware-accelerated PTX instructions, overcoming the historically CPU-bound bottleneck of arithmetic coding.

## AI-Native Codecs

By providing language-level primitives for these operations, Taria allows researchers to build AI-native codecs directly for the GPU. The compiler handles the intricate memory alignment, kernel fusion, and hardware mapping required to run neural encoders and entropy models in real-time, enabling compression targets like **1TB down to 1GB** with minimal fidelity loss.
