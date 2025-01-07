#include <vector> 
#include <random> 
#include <ctime>
#include <chrono>
#include <atomic>
#include <numeric>
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

GradTensor* GradTensor::gaussian(std::vector<size_t> shape, double mean, double stddev, size_t bidx, size_t pidx) {
  // Create a unique seed by combining high-resolution time and a counter
  static std::atomic<unsigned long long> seed_counter{0};

  auto now = std::chrono::high_resolution_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32);

  // Create a generator with the unique seed
  std::mt19937 generator(unique_seed);

  // Create a distribution
  std::normal_distribution<double> distribution(mean, stddev);

  // Calculate total number of elements
  int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

  // Create and fill the vector
  std::vector<double> result(length);
  for (int i = 0; i < length; ++i) {
    result[i] = distribution(generator);
  }

  return new GradTensor(result, shape, bidx, pidx);
}

GradTensor* GradTensor::gaussian_like(GradTensor* input, double mean, double stddev) {
  return GradTensor::gaussian(input->shape(), mean, stddev, input->bidx, input->pidx());
}

GradTensor* GradTensor::uniform(std::vector<size_t> shape, double min, double max, size_t bidx, size_t pidx) {
  // (Use the same unique seeding method as in the gaussian function)
  static std::atomic<unsigned long long> seed_counter{0};

  auto now = std::chrono::high_resolution_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32);

  std::mt19937 generator(unique_seed);
  std::uniform_real_distribution<double> distribution(min, max);

  int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<double> result(length);
  for (int i = 0; i < length; ++i) {
    result[i] = distribution(generator);
  }

  return new GradTensor(result, shape, bidx, pidx);
}

GradTensor* GradTensor::uniform_like(GradTensor* input, double min, double max) {
  return GradTensor::uniform(input->shape(), min, max, input->bidx, input->pidx());
}

GradTensor* GradTensor::ones(std::vector<size_t> shape, size_t bidx, size_t pidx) {
  return new GradTensor(std::vector<double> (CIntegrity::prod(shape), 1.0), shape, bidx, pidx); 
}

GradTensor* GradTensor::ones_like(GradTensor* input) {
  return GradTensor::ones(input->shape(), input->bidx, input->pidx());
}

GradTensor* GradTensor::zeros(std::vector<size_t> shape, size_t bidx, size_t pidx) {
  return new GradTensor(std::vector<double> (CIntegrity::prod(shape), 0.0), shape, bidx, pidx); 
}

GradTensor* GradTensor::zeros_like(GradTensor* input) {
  return GradTensor::zeros(input->shape(), input->bidx, input->pidx());
}
