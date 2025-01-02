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

# Add tests to CTest with improved configuration
include(GoogleTest)
set(DISCOVERY_TIMEOUT 120)  # Increase discovery timeout to 120 seconds
set(TEST_TIMEOUT 300)       # Set individual test timeout to 300 seconds

# Configure test discovery with timeouts and output
gtest_discover_tests(tests
    DISCOVERY_TIMEOUT ${DISCOVERY_TIMEOUT}
    PROPERTIES 
        TIMEOUT ${TEST_TIMEOUT}
        ENVIRONMENT "GTEST_CATCH_EXCEPTIONS=1"
    TEST_PREFIX "tensor_"
    TEST_SUFFIX ""
    NO_PRETTY_TYPES
    NO_PRETTY_VALUES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)

# Add a custom target for running tests directly
add_custom_target(run_tests
    COMMAND tests --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS tests
)
