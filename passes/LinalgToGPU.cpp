#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Pattern to map `taria.gpu_kernel` (now lowered to linalg) into `gpu.launch`.
struct GPULaunchRewrite : public RewritePattern {
    GPULaunchRewrite(MLIRContext *context)
        : RewritePattern("taria.gpu_kernel", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Taria extracts `block_size` and `shared_mem_bytes` attributes.
        // It injects them into the standard MLIR `gpu.launch` structure.
        return failure();
    }
};

struct LinalgToGPUPass : public PassWrapper<LinalgToGPUPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 1. Bufferize tensor semantics to memrefs.
    // 2. Map affine loops to grid/block dimensions using GPU dialect.
    // 3. Promote latency-critical tensors to shared memory (`gpu.subgroup_mma`).
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createLinalgToGPUPass() {
  return std::make_unique<LinalgToGPUPass>();
}
