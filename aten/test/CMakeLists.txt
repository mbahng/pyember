# Configure GoogleTest build
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test files
set(TENSOR_TESTS 
  Tensor/GradTensor/constructors_tests.cpp
  Tensor/GradTensor/util_tests.cpp
  Tensor/GradTensor/algebra_tests.cpp

  Tensor/Tensor/constructors_tests.cpp
  Tensor/Tensor/algebra_tests.cpp
  Tensor/Tensor/grad_tests.cpp
  Tensor/Tensor/math_tests.cpp
)

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

