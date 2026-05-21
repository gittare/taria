cmake_minimum_required(VERSION 3.20)
project(taria_dialects CXX)

# Find MLIR and TableGen
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Setup TableGen
set(LLVM_TARGET_DEFINITIONS Taria/TariaOps.td)
mlir_tablegen(TariaOps.h.inc -gen-op-decls)
mlir_tablegen(TariaOps.cpp.inc -gen-op-defs)
mlir_tablegen(TariaDialect.h.inc -gen-dialect-decls -dialect=taria)
mlir_tablegen(TariaDialect.cpp.inc -gen-dialect-defs -dialect=taria)
mlir_tablegen(TariaEnums.h.inc -gen-enum-decls)

add_public_tablegen_target(TariaOpsIncGen)

add_library(TariaDialects
    Taria/TariaDialect.cpp
)

target_include_directories(TariaDialects PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(TariaDialects TariaOpsIncGen)
target_link_libraries(TariaDialects PRIVATE MLIRIR MLIRInferTypeOpInterface)
