cmake_minimum_required(VERSION 3.20)
project(taria_backend CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Ensure Taria API Headers are in the include path
target_include_directories(TariaDialects PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Find necessary LLVM/MLIR dependencies
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

add_library(TariaBackendCore
    api/BackendAPI.cpp
    compression/SemanticCompressor.cpp
    cuda/CudaRuntime.cpp
    nvptx/PTXOptimizer.cpp
    optimizer/TensorFusion.cpp
)

target_include_directories(TariaBackendCore PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# Link against MLIR core components (required for optimization and dialect use)
target_link_libraries(TariaBackendCore PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRRewrite
)
