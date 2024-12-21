#include <vector> 
#include <ctime>
#include "../Tensor.h"

GradTensor::GradTensor() {
  this->storage_ = std::vector<double>{}; 
  this->shape_ = std::vector<size_t>{}; 
  this-> bidx_ = 0; 
  this->pidx_ = 0;
}

GradTensor::GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t pidx, size_t bidx) { 
  if (data.size() != shape_to_length(shape)) {
    throw std::logic_error("The size of data and the shape do not match.");
  }
  this->storage_ = data; 
  this->shape_ = shape;  
  this->pidx_ = pidx;
  this->bidx_ = bidx; 
}

GradTensor::GradTensor(std::vector<size_t> shape, size_t pidx, size_t bidx) {
  this->storage_ = std::vector<double>(shape_to_length(shape), 0.0); 
  this->shape_ = shape;  
  this->pidx_ = pidx;
  this->bidx_ = bidx;
}

GradTensor* GradTensor::eye(size_t n, size_t pidx, size_t bidx) {
  std::vector<size_t> shape = {n, n}; 
  std::vector<double> storage(n * n, 0.0); 
  for (int i = 0; i < n; i++) {
    storage[n * i + i] = 1.0; 
  }
  return new GradTensor(storage, shape, pidx, bidx);
}

