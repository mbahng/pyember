#include <vector>
#include "Tensor.h"

int shape_to_length(std::vector<size_t> shape);

BaseTensor& BaseTensor::reshape(std::vector<size_t> new_shape) {
  if (shape_to_length(new_shape) != shape_to_length(this->shape_)) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  this->shape_ = new_shape; 
  return *this; 
}

Tensor& Tensor::reshape(std::vector<size_t> new_shape) {
  if (shape_to_length(new_shape) != shape_to_length(this->shape_)) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  this->shape_ = new_shape; 
  return *this; 
}

// include view, split methods
