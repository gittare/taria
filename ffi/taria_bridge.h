// ffi/taria_bridge.h

#pragma once
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#  define TARIA_API __declspec(dllexport)
#else
#  define TARIA_API __attribute__((visibility("default")))
#endif

extern "C" {

struct TariaAstHandle { void* ptr; };

TARIA_API int taria_parse_source(const char* src, TariaAstHandle* out_handle);
TARIA_API void taria_free_ast(TariaAstHandle handle);

// ABI: Only POD types, versioned struct layout
// Allocator: Ownership rules documented, caller frees via taria_free_ast
// Versioning: Add version field to TariaAstHandle for future upgrades
}
