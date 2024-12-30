cmake_minimum_required(VERSION 3.11)
project(aten)
include(FetchContent)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add option 
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_DEBUG "Enable debug build with -g flat" OFF) 
option(BUILD_DEV "Build developerment/testing files" OFF)

if(BUILD_PYTHON_BINDINGS) 
  find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Found Python: ${Python_VERSION}")
endif() 

add_subdirectory(src)

if (BUILD_DEV) 
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG main
  )
  enable_testing()
  add_subdirectory(test)
endif()

if(BUILD_DEBUG) 
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif() 

if(BUILD_PYTHON_BINDINGS) 
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1  
  )
  FetchContent_MakeAvailable(pybind11)
  add_subdirectory(bindings)
endif()

