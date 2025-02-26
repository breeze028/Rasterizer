cmake_minimum_required(VERSION 3.5.0)
project(Rasterizer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(parrallel parrallel.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(parrallel PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(parrallel PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(parrallel PRIVATE /utf-8)
