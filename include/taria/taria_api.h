// include/taria/taria_api.h

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct TariaAstHandle { void* ptr; };

int taria_parse_source(const char* src, TariaAstHandle* out_handle);
void taria_free_ast(TariaAstHandle handle);

#ifdef __cplusplus
}
#endif
