# The Taria MLIR Dialect

The `taria` MLIR dialect is the core intermediate representation bridging the high-level Pythonic DSL semantics and the low-level linear algebra (`linalg`) compute structures.

## Purpose and Design

The primary goal of the `taria` dialect is to preserve the semantics of neural compression pipelines. If we immediately lowered an autoencoder to `linalg.matmul` and `linalg.conv2d`, we would lose the context that the entire sequence is part of a compression phase. Preserving context allows for aggressive, domain-specific optimization passes, such as:

- Codebook caching in shared memory for `vector_quantize`.
- Fusing entropy modeling directly into the quantization loop.
- Dynamic tiling based on compression chunk sizes.

## Operation Definition Specification (ODS)

Taria utilizes MLIR's TableGen (ODS) to declare operations.

### `taria.compress`

A high-level macro operation representing a complete neural compression pipeline.

```tablegen
def Taria_CompressOp : Taria_Op<"compress", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Compresses a tensor into a latent representation.";
  let description = [{
    A macro-operation that fuses neural encoding, vector quantization, and entropy modeling
    into a single optimized GPU pipeline.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$bitstream);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($bitstream)";
}
```

### `taria.vector_quantize`

Maps continuous vectors to the nearest discrete representations.

```tablegen
def Taria_VectorQuantizerOp : Taria_Op<"vector_quantize", [Pure]> {
  let summary = "Performs VQ on a latent vector against a learned codebook.";
  let arguments = (ins AnyTensor:$latent, AnyTensor:$codebook);
  let results = (outs AnyTensor:$quantized, AnyTensor:$indices);
  let assemblyFormat = "$latent `,` $codebook attr-dict `:` type($latent) `,` type($codebook) `->` type($quantized) `,` type($indices)";
}
```

## Rewrite Patterns and Lowering

The Taria compiler uses the MLIR `GreedyPatternRewriteDriver` to fold and optimize operations before lowering them to standard dialects.

For example, the conversion from `taria` to `linalg` is handled in `TariaToLinalg.cpp`:

```cpp
struct EncodeToLinalgRewrite : public RewritePattern {
    EncodeToLinalgRewrite(MLIRContext *context)
        : RewritePattern("taria.encode", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Pseudo-code for expanding `taria.encode` into Linalg ops
        Value input = op->getOperand(0);

        // 1. Bufferize tensor to memref
        // 2. Emit linalg.conv_2d
        // 3. Emit linalg.generic for activation

        return success();
    }
};
```

## Custom Types

Taria enforces strict tensor dimension checking at the MLIR level. The frontend guarantees shape validity, but MLIR verification rules act as a secondary safety net. Future custom types will include `!taria.latent_space` and `!taria.codebook` to provide stricter alias analysis during the `LinalgToGPU` conversion pass.
