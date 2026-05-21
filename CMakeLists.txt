cmake_minimum_required(VERSION 3.20)
project(taria LANGUAGES CXX)

# --- Configuration & Flags ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization, LTO, Sanitizers (optional but recommended in debug/dev)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

# --- Dependencies ---
# Mocking LLVM/MLIR package configs for bootstrap
# In a real setup, provide CMAKE_PREFIX_PATH to a built LLVM/MLIR tree.
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

include_directories(include)

# --- Subdirectories ---
add_subdirectory(dialects)
add_subdirectory(passes)
add_subdirectory(backend)
add_subdirectory(runtime)
