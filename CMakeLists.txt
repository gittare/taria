cmake_minimum_required(VERSION 3.20)
project(taria LANGUAGES CXX CUDA)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CUDA REQUIRED)

add_subdirectory(dialects)
add_subdirectory(passes)
add_subdirectory(backend)
add_subdirectory(runtime)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -fsanitize=address,undefined")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3")

# Shared/static libs, install rules, cross-platform support
