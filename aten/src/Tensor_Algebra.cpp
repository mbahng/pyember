#include <vector>
#include <cassert>
#include "Tensor.h"
// Functions to support operations in Tensor Algebras and 
// Noncommutative Matrix Rings. Includes: 
// - Addition 
// - Scalar Multiplication 
// - Matrix Multiplication

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

Tensor Tensor::sub(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::scamul(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}


