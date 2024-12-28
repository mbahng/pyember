cmake_minimum_required(VERSION 3.10)

# Find Python
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)

# Enable testing
enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List all source files except main.cpp
set(TENSOR_SOURCES
  src/Util/utils.cpp

  src/Tensor/BaseTensor/base.cpp
  src/Tensor/BaseTensor/comparison.cpp
  src/Tensor/BaseTensor/index.cpp
  src/Tensor/BaseTensor/string.cpp

  src/Tensor/GradTensor/algebra.cpp
  src/Tensor/GradTensor/base.cpp
  src/Tensor/GradTensor/comparison.cpp
  src/Tensor/GradTensor/constructor.cpp
  src/Tensor/GradTensor/index.cpp
  src/Tensor/GradTensor/math.cpp
  src/Tensor/GradTensor/shape.cpp
  src/Tensor/GradTensor/string.cpp

  src/Tensor/Tensor/algebra.cpp
  src/Tensor/Tensor/backprop.cpp
  src/Tensor/Tensor/base.cpp
  src/Tensor/Tensor/comparison.cpp
  src/Tensor/Tensor/constructor.cpp
  src/Tensor/Tensor/index.cpp
  src/Tensor/Tensor/math.cpp
  src/Tensor/Tensor/shape.cpp
  src/Tensor/Tensor/string.cpp
)
set(TENSOR_TESTS 
  test/Tensor/GradTensor/constructors_tests.cpp
  test/Tensor/GradTensor/util_tests.cpp
  test/Tensor/GradTensor/algebra_tests.cpp

  test/Tensor/Tensor/constructors_tests.cpp
  test/Tensor/Tensor/algebra_tests.cpp
  test/Tensor/Tensor/grad_tests.cpp
  test/Tensor/Tensor/math_tests.cpp
)

# Create the main library
add_library(aten ${TENSOR_SOURCES})
target_include_directories(aten 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Create the main executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE aten)

# Set up Google Test with specific options for ARM64
set(CMAKE_OSX_ARCHITECTURES "arm64")
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

# Configure GoogleTest build
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

# Set up pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1  # Use the latest stable version
)

# Make dependencies available
FetchContent_MakeAvailable(googletest pybind11)

# Create test executable
add_executable(tests ${TENSOR_TESTS})
target_include_directories(tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against GTest and your library
target_link_libraries(tests
    PRIVATE
        aten
        gtest
        gtest_main
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(tests)

# Python bindings option
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
