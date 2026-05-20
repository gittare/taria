// passes/GPUToNVVM.cpp

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GPUToNVVMPass : public PassWrapper<GPUToNVVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Lower gpu ops to NVVM dialect
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createGPUToNVVMPass() {
  return std::make_unique<GPUToNVVMPass>();
}

static PassRegistration<GPUToNVVMPass> pass("gpu-to-nvvm", "Lower GPU ops to NVVM");
