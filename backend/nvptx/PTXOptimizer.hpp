#pragma once

#include <string>
#include <vector>

namespace mlir { class ModuleOp; }

namespace taria {
namespace nvptx {

/// High-level PTX Optimization and Emission Engine
///
/// Translates heavily optimized LLVM IR (generated from the MLIR `nvvm` dialect)
/// into bare-metal PTX assembly.
///
/// Features:
/// - LLVM TargetMachine invocation (sm_80, sm_90).
/// - Loop unrolling and FMA instruction combining.
/// - Warp-level register allocation overrides.
class PTXOptimizer {
public:
    PTXOptimizer(const std::string& target_architecture = "sm_80");

    /// Compiles an MLIR Module (already lowered to LLVM dialect) to a PTX string.
    std::string compile_to_ptx(mlir::ModuleOp module);

private:
    std::string target_arch_;

    // Internal method to run LLVM O3 passes specific to NVPTX
    void run_llvm_passes(void* llvm_module);
};

} // namespace nvptx
} // namespace taria
