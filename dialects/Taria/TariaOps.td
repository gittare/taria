//===- TariaOps.td - Taria dialect ops ---------------------*- tablegen -*-===//
//
// Defines the operations of the Taria dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TARIA_OPS
#define TARIA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Taria_Dialect : Dialect {
  let name = "taria";
  let cppNamespace = "::mlir::taria";
  let summary = "The Taria dialect for semantic compression and GPU tensor ops.";
  let description = [{
    The Taria dialect is designed to represent semantic compression operations,
    latent vectors, and high-level GPU concepts before they are lowered to
    Linalg or standard GPU dialects.
  }];
}

class Taria_Op<string mnemonic, list<Trait> traits = []> :
    Op<Taria_Dialect, mnemonic, traits>;

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

def Taria_EntropyModelOp : Taria_Op<"entropy_model", [Pure]> {
  let summary = "Applies a learned entropy model to a latent tensor.";
  let description = [{
    The `entropy_model` operation estimates the probability distribution of
    the quantized latents to generate a highly compressed bitstream. It is
    a cornerstone of semantic AI compression pipelines.
  }];
  let arguments = (ins AnyTensor:$latent, AnyTensor:$cdf);
  let results = (outs AnyTensor:$bitstream);
  let assemblyFormat = "$latent `,` $cdf attr-dict `:` type($latent) `,` type($cdf) `->` type($bitstream)";
}

def Taria_VectorQuantizerOp : Taria_Op<"vector_quantize", [Pure]> {
  let summary = "Performs VQ on a latent vector against a learned codebook.";
  let description = [{
    The `vector_quantize` operation maps continuous vectors to the nearest
    discrete representations found in the provided codebook tensor.
  }];
  let arguments = (ins AnyTensor:$latent, AnyTensor:$codebook);
  let results = (outs AnyTensor:$quantized, AnyTensor:$indices);
  let assemblyFormat = "$latent `,` $codebook attr-dict `:` type($latent) `,` type($codebook) `->` type($quantized) `,` type($indices)";
}

def Taria_TensorTilingOp : Taria_Op<"tensor_tile", [Pure]> {
  let summary = "Declares an explicit tile map for semantic data chunking.";
  let arguments = (ins AnyTensor:$input, I64ArrayAttr:$tile_sizes);
  let results = (outs AnyTensor:$tiled);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($tiled)";
}

def Taria_EncodeOp : Taria_Op<"encode", [Pure]> {
  let summary = "Encodes a tensor using an autoencoder model.";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$latent);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($latent)";
}

def Taria_DecodeOp : Taria_Op<"decode", [Pure]> {
  let summary = "Decodes a latent tensor back to the original space.";
  let arguments = (ins AnyTensor:$latent);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$latent attr-dict `:` type($latent) `->` type($output)";
}

def Taria_QuantizeOp : Taria_Op<"quantize", [Pure]> {
  let summary = "Quantizes a latent tensor.";
  let arguments = (ins AnyTensor:$latent);
  let results = (outs AnyTensor:$quantized);
  let assemblyFormat = "$latent attr-dict `:` type($latent) `->` type($quantized)";
}

def Taria_GPUKernelOp : Taria_Op<"gpu_kernel", [IsolatedFromAbove]> {
  let summary = "Defines a Taria GPU kernel with explicit blocking and shared memory constraints.";
  let description = [{
    The `gpu_kernel` operation wraps a region of compute designed for direct NVPTX generation.
    It carries attributes for `block_size` and `shared_mem_bytes` to inform occupancy
    calculations and auto-tiling optimizations.
  }];
  let arguments = (ins I32Attr:$block_size, I32Attr:$shared_mem_bytes);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`block_size` `(` $block_size `)` `shared_mem_bytes` `(` $shared_mem_bytes `)` $body attr-dict";
}

def Taria_ReturnOp : Taria_Op<"return", [Terminator]> {
  let summary = "Return operation for Taria kernel regions.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

#endif // TARIA_OPS
