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
  src/BaseTensor/integrity.cpp
  src/BaseTensor/io.cpp
  src/BaseTensor/util.cpp
  src/BaseTensor/view.cpp
  src/Tensor/constructors.cpp
  src/Tensor/algebra.cpp
  src/Tensor/grad.cpp
  src/Tensor/math.cpp
  src/Tensor/util.cpp
  src/GradTensor/constructors.cpp
  src/GradTensor/algebra.cpp
  src/GradTensor/util.cpp
)
set(TENSOR_TESTS 
  test/BaseTensor/integrity_tests.cpp
  test/BaseTensor/io_tests.cpp
  test/BaseTensor/util_tests.cpp
  test/BaseTensor/view_tests.cpp
  test/Tensor/constructors_tests.cpp
  test/Tensor/algebra_tests.cpp
  test/Tensor/grad_tests.cpp
  test/Tensor/math_tests.cpp
  test/GradTensor/constructors_tests.cpp
  test/GradTensor/util_tests.cpp
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
