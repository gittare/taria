#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace taria {
namespace optimizer {

/// MLIR Pass representing the Taria Tensor Fusion Engine.
///
/// Purpose:
/// Locates adjacent data-parallel operations (e.g. taria.encode -> taria.quantize)
/// and fuses them into a `taria.tensor.fuse` region. This prevents intermediate
/// buffer materialization (VRAM read/write), keeping latents in registers/SRAM.
///
/// Performance:
/// Heavily relies on MLIR's GreedyPatternRewriteDriver. Memory coalescing and
/// register-pressure constraints are checked before applying the fusion.
class TensorFusionPass : public mlir::PassWrapper<TensorFusionPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorFusionPass)

    void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createTensorFusionPass();

} // namespace optimizer
} // namespace taria
