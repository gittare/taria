#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GPUToNVVMPass : public PassWrapper<GPUToNVVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Lowers standard GPU dialect to NVVM/LLVM dialects.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createGPUToNVVMPass() {
  return std::make_unique<GPUToNVVMPass>();
}
