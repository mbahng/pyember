# should be >= 3.15 since it prioritizes finding python in current venv
cmake_minimum_required(VERSION 3.15)
project(aten)
include(FetchContent)

# Set C++ standard and have it support both architectures
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")

if(UNIX AND NOT APPLE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_compile_options(-fPIC)
endif()

# add option 
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_DEBUG "Enable debug build with -g flag" ON) 
option(BUILD_DEV "Build developerment/testing files" ON)

add_subdirectory(src)

if (BUILD_DEV) 
  message(STATUS "Development mode ON")
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG main
  )
  enable_testing()
  add_subdirectory(test)
else() 
  message(STATUS "Development mode OFF")
endif()

if(BUILD_DEBUG)
    message(STATUS "Debug mode ON")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
else()
    message(STATUS "Debug mode OFF")
endif()

if(BUILD_PYTHON_BINDINGS) 
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
  set(Python3_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
  set(Python3_LIBRARY ${PYTHON_LIBRARY})
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13
  )
  FetchContent_MakeAvailable(pybind11)
  add_subdirectory(bindings)
endif()

