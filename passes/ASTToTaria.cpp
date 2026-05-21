#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Note: In a complete compiler, this pass bridges the FFI and MLIR structures.
// It matches Rust AST nodes injected into MLIR attributes and expands them.

using namespace mlir;

namespace {
struct ASTToTariaPass : public PassWrapper<ASTToTariaPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 1. Iterate over all generic `func.func` ops.
    // 2. Identify `@gpu.kernel` attributes attached by the FFI.
    // 3. Rewrite them into `taria.gpu_kernel` operations containing nested tensor ops.
    //
    // This is the semantic bridging layer.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createASTToTariaPass() {
  return std::make_unique<ASTToTariaPass>();
}
