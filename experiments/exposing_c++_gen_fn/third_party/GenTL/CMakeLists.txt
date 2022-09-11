cmake_minimum_required(VERSION 3.12...3.22)

project(GenTL VERSION 1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Test
include(CTest)
include(FetchContent)
add_subdirectory(tests)

# Docs
find_package(Doxygen)
if(Doxygen_FOUND)
    add_subdirectory(docs)
else()
    message(STATUS "Doxygen not found, not building docs")
endif()

# Examples
add_subdirectory(examples)
