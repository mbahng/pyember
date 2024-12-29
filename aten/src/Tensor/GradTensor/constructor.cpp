#include <vector> 
#include <ctime>
#include "../Tensor.h"
#include "../../Util/utils.h"

GradTensor::GradTensor(double scalar) {
  this->_storage = std::vector<double>{scalar}; 
  this->_shape = std::vector<size_t>{1}; 
  this->bidx = 0; 
  this->_bshape = std::vector<size_t>{}; 
  this->_nbshape = std::vector<size_t>{1}; 
  this->_pidx = 0;
  this->_size = 1; 
  this->_rank = 0; 
}

GradTensor::GradTensor(std::vector<double> storage, std::vector<size_t> shape, size_t bidx, size_t pidx) { 
  if (storage.size() != CIntegrity::prod(shape)) {
    throw std::logic_error("The size of storage and the shape do not match.");
  }
  this->_storage = storage; 
  this->_shape = shape;  
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->_pidx = pidx; 
  this->_size = CIntegrity::prod(shape); 
  this->_rank = shape.size(); 
}

GradTensor::GradTensor(std::vector<size_t> shape, size_t bidx, size_t pidx) {
  this->_storage = std::vector<double>(CIntegrity::prod(shape), 0.0); 
  this->_shape = shape;  
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->_pidx = pidx;
  this->_size = CIntegrity::prod(shape); 
  this->_rank = shape.size(); 
}

GradTensor* GradTensor::eye(size_t n, size_t bidx, size_t pidx) {
  std::vector<size_t> shape = {n, n}; 
  std::vector<double> storage(n * n, 0.0); 
  for (int i = 0; i < n; i++) {
    storage[n * i + i] = 1.0; 
  }
  return new GradTensor(storage, shape, bidx, pidx);
}

