cmake_minimum_required(VERSION 3.4...3.18)
project(ember_tensor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Add your extension module
pybind11_add_module(tensor_cpp tensor.cpp tensor_binding.cpp)

# Optionally, you can add include directories if needed
# target_include_directories(tensor_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# For ARM64 specific flags (if needed)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(tensor_cpp PRIVATE -march=armv8-a)
endif()

