// dialects/Taria/TariaDialect.td

def Taria_Dialect : Dialect {
  let name = "taria";
  let cppNamespace = "::mlir::taria";
}

def Taria_CompressOp : Op<"compress", [NoSideEffect]> {
  let summary = "Semantic compression op";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `->` type($output)";
}

def Taria_EncodeOp : Op<"encode", [NoSideEffect]> { ... }
def Taria_DecodeOp : Op<"decode", [NoSideEffect]> { ... }
def Taria_LatentOp : Op<"latent", [NoSideEffect]> { ... }
def Taria_GPUKernelOp : Op<"gpu_kernel", [AttrSizedOperandSegments]> { ... }

// Canonicalization: Provide hooks in C++
// Type inference: Implement inferReturnTypes
// Verification: Implement verify() in C++
// Dialect interfaces: Register with MLIRContext
