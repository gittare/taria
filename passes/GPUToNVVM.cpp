#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GPUToNVVMPass : public PassWrapper<GPUToNVVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 1. Lower standard `gpu` ops to `nvvm` (NVIDIA) or `rocdl` (AMD).
    // 2. Perform register alignment passes.
    // 3. Emit LLVM IR via `LLVMTranslationDialectInterface`.
    //
    // This is the final step before handing off the IR string to the LLVM TargetMachine
    // for PTX / ISA emission.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createGPUToNVVMPass() {
  return std::make_unique<GPUToNVVMPass>();
}
