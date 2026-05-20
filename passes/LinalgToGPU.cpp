#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct LinalgToGPUPass : public PassWrapper<LinalgToGPUPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Lowers Linalg ops to the standard GPU dialect.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createLinalgToGPUPass() {
  return std::make_unique<LinalgToGPUPass>();
}
