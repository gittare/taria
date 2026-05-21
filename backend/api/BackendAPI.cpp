#include "BackendAPI.hpp"

// #include "mlir/IR/MLIRContext.h"
// #include "mlir/Pass/PassManager.h"

extern "C" {

TariaCompilerContext* taria_backend_init() {
    auto* ctx = new TariaCompilerContext();

    // ctx->mlir_context = new mlir::MLIRContext();
    // ctx->pass_manager = new mlir::PassManager((mlir::MLIRContext*)ctx->mlir_context);

    return ctx;
}

void taria_backend_destroy(TariaCompilerContext* ctx) {
    if (ctx) {
        // delete (mlir::PassManager*)ctx->pass_manager;
        // delete (mlir::MLIRContext*)ctx->mlir_context;
        delete ctx;
    }
}

const char* taria_backend_compile_ast_to_ptx(TariaCompilerContext* ctx, void* opaque_ast_handle) {
    // 1. Convert opaque_ast_handle -> taria MLIR dialect
    // 2. Run mlir::PassManager (Fusion, Bufferization, LinalgToGPU)
    // 3. Emit PTX string

    static const char* mock_ptx = ".version 7.5\n// Compiled Taria Kernel";
    return mock_ptx;
}

} // extern "C"
