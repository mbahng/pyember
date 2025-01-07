#include <vector> 
#include <random> 
#include <ctime>
#include <chrono>
#include <atomic>
#include <numeric>
#include "../Tensor.h" 
#include "../../Util/utils.h"

Tensor::Tensor(double scalar, bool requires_grad) {
  this->_storage = std::vector<double>{scalar}; 
  this->_shape = std::vector<size_t>{1}; 
  this->bidx = 0;  
  this->_bshape = std::vector<size_t>{}; 
  this->_nbshape = std::vector<size_t>{}; 
  this->requires_grad = requires_grad; 
  this->_rank = 0; 
}

Tensor::Tensor(std::vector<size_t> shape, size_t bidx, bool requires_grad) {
  this->_storage = std::vector<double>(CIntegrity::prod(shape), 0.0); 
  this->_shape = shape; 
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->requires_grad = requires_grad; 
  this->_rank = shape.size(); 
}

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape, size_t bidx, bool requires_grad) {
  this->_storage = data; 
  this->_shape = shape;  
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->requires_grad = requires_grad; 
  this->_rank = shape.size(); 
}

Tensor::Tensor(std::vector<double> data, size_t bidx, bool requires_grad) {
  this->_storage = data; 
  std::vector<size_t> shape = {data.size()};
  this->_shape = shape; 
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->requires_grad = requires_grad; 
  this->_rank = shape.size();
}

Tensor::Tensor(std::vector<std::vector<double>> data, size_t bidx, bool requires_grad) {
  std::vector<size_t> shape = {data.size(), data[0].size()};
  CIntegrity::array_matches_shape(data, shape); 
  this->_shape = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    res.insert(res.end(), data[i].begin(), data[i].end()); 
  }
  this->_storage = res;  
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->requires_grad = requires_grad; 
  this->_rank = shape.size();
}

Tensor::Tensor(std::vector<std::vector<std::vector<double>>> data, size_t bidx, bool requires_grad) {
  std::vector<size_t> shape = {data.size(), data[0].size(), data[0][0].size()}; 
  CIntegrity::array_matches_shape(data, shape); 
  this->_shape = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      res.insert(res.end(), data[i][j].begin(), data[i][j].end()); 
    }
  }
  this->_storage = res;  
  this->bidx = bidx; 
  this->_bshape = std::vector<size_t>(_shape.begin(), _shape.begin() + bidx);
  this->_nbshape = std::vector<size_t>(_shape.begin() + bidx, _shape.end());
  this->requires_grad = requires_grad; 
  this->_rank = shape.size();
}

Tensor* Tensor::arange(int start, int stop, int step, bool requires_grad) {
  std::vector<double> _storage = {}; 
  for (int i = start; i < stop; i += step) {
    _storage.push_back(i);
  }
  return new Tensor(_storage, std::vector<size_t>{_storage.size()}, 0, requires_grad);
}

Tensor* Tensor::linspace(double start, double stop, int numsteps, bool requires_grad){
  std::vector<double> _storage = {}; 
  double stepsize = (stop - start) / (numsteps - 1); 
  for (double i = start; i <= stop; i += stepsize) {
    _storage.push_back(i);
  }
  return new Tensor(_storage, std::vector<size_t>{_storage.size()}, 0, requires_grad);
}

Tensor* Tensor::gaussian(std::vector<size_t> shape, double mean, double stddev, size_t bidx, bool requires_grad) {
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

  return new Tensor(result, shape, bidx, requires_grad);
}

Tensor* Tensor::gaussian_like(Tensor* input, double mean, double stddev) {
  return Tensor::gaussian(input->shape(), mean, stddev, input->bidx, input->requires_grad);
}

Tensor* Tensor::uniform(std::vector<size_t> shape, double min, double max, size_t bidx, bool requires_grad) {
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

  return new Tensor(result, shape, bidx, requires_grad);
}

Tensor* Tensor::uniform_like(Tensor* input, double min, double max) {
  return Tensor::uniform(input->shape(), min, max, input->bidx, input->requires_grad);
}

Tensor* Tensor::ones(std::vector<size_t> shape, size_t bidx, bool requires_grad) {
  return new Tensor(std::vector<double> (CIntegrity::prod(shape), 1.0), shape, bidx, requires_grad); 
}

Tensor* Tensor::ones_like(Tensor* input) {
  return Tensor::ones(input->shape(), input->bidx, input->requires_grad);
}

Tensor* Tensor::zeros(std::vector<size_t> shape, size_t bidx, bool requires_grad) {
  return new Tensor(std::vector<double>(CIntegrity::prod(shape), 0.0), shape, bidx, requires_grad); 
}

Tensor* Tensor::zeros_like(Tensor* input) {
  return Tensor::zeros(input->shape(), input->bidx, input->requires_grad);
}

