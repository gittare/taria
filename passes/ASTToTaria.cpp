#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct ASTToTariaPass : public PassWrapper<ASTToTariaPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Conceptually translates the frontend AST into the Taria dialect.
    // In practice this might be an MLIR Gen step directly from Rust,
    // or an importer in C++ taking the opaque AST handle.
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createASTToTariaPass() {
  return std::make_unique<ASTToTariaPass>();
}
