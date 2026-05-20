#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TariaToLinalgPass : public PassWrapper<TariaToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Lowers Taria tensor/compression operations to Linalg generics.
    // e.g. taria.encode -> linalg.generic
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createTariaToLinalgPass() {
  return std::make_unique<TariaToLinalgPass>();
}
