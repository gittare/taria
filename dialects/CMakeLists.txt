cmake_minimum_required(VERSION 3.20)
project(taria_dialects CXX)

add_library(TariaDialects
    Taria/TariaDialect.cpp
)

# target_link_libraries(TariaDialects PRIVATE MLIRIR)
