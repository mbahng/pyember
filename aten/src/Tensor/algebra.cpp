#include <vector>
#include <cassert>
#include "../Tensor.h"

int shape_to_length(std::vector<size_t> shape); 

void array_matches_shape(
  std::vector<double> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
);

Tensor Tensor::add(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return Tensor(res_data, this->shape());   
}

Tensor Tensor::add(GradTensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return Tensor(res_data, this->shape());   
}

Tensor Tensor::sub(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::sub(GradTensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::mul(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::mul(GradTensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::matmul(Tensor& other) {
  // Check if the tensors are at least 2D
  assert(this->shape().size() == 2 || other.shape().size() == 2);

  // Check if the last dimension of this tensor matches the second-to-last dimension of other
  assert(this->shape()[1] == other.shape()[0]);

  // Determine the dimensions of the result
  std::vector<size_t> result_shape {this->shape()[0], other.shape()[1]};

  Tensor out(std::vector<double> (shape_to_length(result_shape), 0.0), result_shape);

  // Perform batch matrix multiplication
  int m = this->shape()[0];
  int n = this->shape()[1];
  int p = other.shape()[1];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += this->data()[i * n + k] * other.data()[k * p + j];
      }
      out.storage_[i * p + j] = sum;
    }
  }
  return out;
}


