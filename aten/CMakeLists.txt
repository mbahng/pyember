cmake_minimum_required(VERSION 3.11)
project(aten)
include(FetchContent)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add option 
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_DEBUG "Enable debug build with -g flag" ON) 
option(BUILD_DEV "Build developerment/testing files" ON)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Found Python: ${Python_VERSION}")
  message(STATUS "Python Executable: ${Python_EXECUTABLE}")
endif()

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
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13
  )
  FetchContent_MakeAvailable(pybind11)
  add_subdirectory(bindings)
endif()

