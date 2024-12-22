#include <vector> 
#include <ctime>
#include "../Tensor.h"
#include "../utils.h"

GradTensor::GradTensor() {
  this->storage_ = std::vector<double>{}; 
  this->shape_ = std::vector<size_t>{}; 
  this->bidx_ = 0; 
  this->pidx_ = 0;
}

GradTensor::GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t bidx, size_t pidx) { 
  if (data.size() != shape_to_length(shape)) {
    throw std::logic_error("The size of data and the shape do not match.");
  }
  this->storage_ = data; 
  this->shape_ = shape;  
  this->bidx_ = bidx; 
  this->pidx_ = pidx;
}

GradTensor::GradTensor(std::vector<size_t> shape, size_t bidx, size_t pidx) {
  this->storage_ = std::vector<double>(shape_to_length(shape), 0.0); 
  this->shape_ = shape;  
  this->bidx_ = bidx; 
  this->pidx_ = pidx;
}

GradTensor* GradTensor::eye(size_t n, size_t bidx, size_t pidx) {
  std::vector<size_t> shape = {n, n}; 
  std::vector<double> storage(n * n, 0.0); 
  for (int i = 0; i < n; i++) {
    storage[n * i + i] = 1.0; 
  }
  return new GradTensor(storage, shape, bidx, pidx);
}

