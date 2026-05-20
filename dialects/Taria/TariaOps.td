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

def Taria_CompressOp : Taria_Op<"compress", [Pure]> {
  let summary = "Compresses a tensor into a latent representation.";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$latent);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($latent)";
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

def Taria_GPUKernelOp : Taria_Op<"gpu_kernel", []> {
  let summary = "Defines a Taria GPU kernel.";
  let regions = (region AnyRegion:$body);
  // simplified for bootstrapping
}

def Taria_ReturnOp : Taria_Op<"return", [Terminator]> {
  let summary = "Return operation for Taria kernel regions.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

#endif // TARIA_OPS
