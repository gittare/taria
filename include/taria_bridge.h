#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#  define TARIA_API __declspec(dllexport)
#else
#  define TARIA_API __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * @brief Opaque handle to the Taria AST
 */
struct TariaAstHandle {
    void* ptr;
};

/**
 * @brief Parses Taria source code into an AST.
 *
 * @param src Null-terminated UTF-8 C string containing the source code.
 * @param out_handle Pointer to write the resulting AST handle to.
 * @return 0 on success, negative error code on failure.
 */
TARIA_API int taria_parse_source(const char* src, TariaAstHandle* out_handle);

/**
 * @brief Frees an AST handle.
 *
 * @param handle The handle to free.
 */
TARIA_API void taria_free_ast(TariaAstHandle handle);

} // extern "C"
