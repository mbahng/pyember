cmake_minimum_required(VERSION 3.10)
project(aten)

# add option to python bind, default on 
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python: ${Python_VERSION}")

add_subdirectory(aten)

