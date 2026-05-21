#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

// Pretend we have Taria Dialect and Linalg headers
// #include "TariaDialect.h"
// #include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

/// Pattern to convert `taria.encode` to `linalg.generic` (e.g. MatMul / Conv sequences).
struct EncodeToLinalgRewrite : public RewritePattern {
    EncodeToLinalgRewrite(MLIRContext *context)
        : RewritePattern("taria.encode", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Taria neural encoders are lowered to a series of Linalg convolutions and matmuls.
        // Concept:
        // Value input = op->getOperand(0);
        // auto linalg_matmul = rewriter.create<linalg::MatmulOp>(op->getLoc(), ...);
        // rewriter.replaceOp(op, linalg_matmul.getResults());
        return failure(); // Stub
    }
};

struct TariaToLinalgPass : public PassWrapper<TariaToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Configure target: illegalize high-level semantic compression ops
    target.addLegalDialect<BuiltinDialect>();
    target.addIllegalOp<RewritePattern>(); // Fake for demo

    RewritePatternSet patterns(context);
    patterns.add<EncodeToLinalgRewrite>(context);

    // Apply partial conversion
    // if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    //     signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createTariaToLinalgPass() {
  return std::make_unique<TariaToLinalgPass>();
}
