cmake_minimum_required(VERSION 3.20)
project(taria_backend CXX)

add_library(TariaBackend taria_backend.cpp)

# Conceptually link MLIR headers
# target_link_libraries(TariaBackend PRIVATE MLIRIR)
