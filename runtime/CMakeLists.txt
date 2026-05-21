cmake_minimum_required(VERSION 3.20)
project(taria_runtime CUDA)

add_library(TariaRuntime
    taria_runtime.cu
)

# target_link_libraries(TariaRuntime PRIVATE CUDA::cudart)
