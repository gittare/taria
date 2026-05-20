// passes/TariaToLinalg.cpp

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TariaToLinalgPass : public PassWrapper<TariaToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Pattern rewriter: match taria.compress, rewrite to linalg ops
    // Greedy rewrite: apply patterns until fixpoint
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createTariaToLinalgPass() {
  return std::make_unique<TariaToLinalgPass>();
}

// Register pass with MLIR
static PassRegistration<TariaToLinalgPass> pass("taria-to-linalg", "Lower Taria ops to Linalg");
