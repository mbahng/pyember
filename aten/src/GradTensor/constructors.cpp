#include <vector> 
#include <ctime>
#include "../Tensor.h"

// Grad Tensors

GradTensor::GradTensor() {
  this->storage_ = std::vector<double>{}; 
  this->shape_ = std::vector<size_t>{};  
  this->pivot_ = 0;
}

GradTensor::GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t pivot) {
  this->storage_ = data; 
  this->shape_ = shape;  
  this->pivot_ = pivot;
}

GradTensor::GradTensor(std::vector<size_t> shape, size_t pivot) {
  this->storage_ = std::vector<double>(shape_to_length(shape), 0.0); 
  this->shape_ = shape;  
  this->pivot_ = pivot;
}

GradTensor GradTensor::eye(size_t n, size_t pivot) {
  std::vector<size_t> shape = {n, n}; 
  std::vector<double> storage(n * n, 0.0); 
  for (int i = 0; i < n * n; i++) {
    storage[n * i + i] = 1.0; 
  }
  return GradTensor(storage, shape, pivot);
}

