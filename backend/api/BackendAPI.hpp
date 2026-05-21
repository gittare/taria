#pragma once

// Taria Backend C-API for FFI Integration
// This acts as the stable ABI boundary for the Rust frontend to invoke compilation
// and execution phases.

#ifdef _WIN32
#  define TARIA_BACKEND_API __declspec(dllexport)
#else
#  define TARIA_BACKEND_API __attribute__((visibility("default")))
#endif

extern "C" {

struct TariaCompilerContext {
    void* mlir_context;
    void* pass_manager;
};

/// Initializes the backend compilation environment (MLIR contexts, passes, target setup).
TARIA_BACKEND_API TariaCompilerContext* taria_backend_init();

/// Frees the backend compiler context.
TARIA_BACKEND_API void taria_backend_destroy(TariaCompilerContext* ctx);

/// Compiles an AST module string into PTX. (Mocked signature).
TARIA_BACKEND_API const char* taria_backend_compile_ast_to_ptx(TariaCompilerContext* ctx, void* opaque_ast_handle);

} // extern "C"
