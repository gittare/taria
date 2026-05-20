// passes/LinalgToGPU.cpp

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct LinalgToGPUPass : public PassWrapper<LinalgToGPUPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Lower linalg ops to gpu.launch, gpu.memcpy, etc.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createLinalgToGPUPass() {
  return std::make_unique<LinalgToGPUPass>();
}

static PassRegistration<LinalgToGPUPass> pass("linalg-to-gpu", "Lower Linalg ops to GPU");
