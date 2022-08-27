cmake_minimum_required(VERSION 3.12...3.22)

project(extern_gen_fn VERSION 1.0 LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
message(STATUS "Building for CPU, no CUDA")

find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_library(gen_fn_impl lib/gen_fn.cpp)
target_include_directories(gen_fn_impl PRIVATE include)
target_include_directories(gen_fn_impl PUBLIC third_party/GenTL/include)

pybind11_add_module(extern_gen_fn ${CMAKE_CURRENT_LIST_DIR}/lib/main.cpp)
target_include_directories(extern_gen_fn PRIVATE include)
target_include_directories(extern_gen_fn PUBLIC third_party/GenTL/include)
target_link_libraries(extern_gen_fn PRIVATE gen_fn_impl)
