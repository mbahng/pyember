# source files
set(TENSOR_SOURCES
  Util/utils.cpp

  Tensor/BaseTensor/base.cpp
  Tensor/BaseTensor/comparison.cpp
  Tensor/BaseTensor/index.cpp
  Tensor/BaseTensor/string.cpp

  Tensor/GradTensor/algebra.cpp
  Tensor/GradTensor/base.cpp
  Tensor/GradTensor/comparison.cpp
  Tensor/GradTensor/constructor.cpp
  Tensor/GradTensor/index.cpp
  Tensor/GradTensor/math.cpp
  Tensor/GradTensor/shape.cpp
  Tensor/GradTensor/string.cpp

  Tensor/Tensor/algebra.cpp
  Tensor/Tensor/backprop.cpp
  Tensor/Tensor/base.cpp
  Tensor/Tensor/comparison.cpp
  Tensor/Tensor/constructor.cpp
  Tensor/Tensor/index.cpp
  Tensor/Tensor/math.cpp
  Tensor/Tensor/shape.cpp
  Tensor/Tensor/string.cpp
)

# Create the main library
add_library(aten ${TENSOR_SOURCES})
target_include_directories(aten 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Create the main executable
add_executable(main ../main.cpp)
target_link_libraries(main PRIVATE aten)

