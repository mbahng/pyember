cmake_minimum_required(VERSION 3.10)
project(aten)

# Enable testing
enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List all source files except main.cpp
set(TENSOR_SOURCES
    src/Tensor_Algebra.cpp
    src/Tensor_Constructors.cpp
    src/Tensor_IO.cpp
    src/Tensor_Index.cpp
    src/Tensor_Integrity.cpp
    src/Tensor_Math.cpp
    src/Tensor_Util.cpp
    src/Tensor_View.cpp
)

set(TENSOR_TESTS 
    test/Tensor_Constructors_test.cpp
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

# Make GoogleTest available
FetchContent_MakeAvailable(googletest)

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
