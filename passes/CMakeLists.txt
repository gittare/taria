cmake_minimum_required(VERSION 3.20)
project(taria_passes CXX)

add_library(TariaPasses
    ASTToTaria.cpp
    GPUToNVVM.cpp
    LinalgToGPU.cpp
    TariaToLinalg.cpp
)

# target_link_libraries(TariaPasses PRIVATE MLIRIR)
