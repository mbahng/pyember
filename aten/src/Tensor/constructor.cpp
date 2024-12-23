#include <vector> 
#include <random> 
#include <ctime>
#include "../Tensor.h" 
#include "../utils.h"

Tensor::Tensor(std::vector<size_t> shape, size_t bidx, bool has_grad) {
  this->storage_ = std::vector<double>(CIntegrity::prod(shape), 0.0); 
  this->shape_ = shape; 
  this->bidx_ = bidx; 
  this->bshape_ = std::vector<size_t>(shape_.begin(), shape_.begin() + bidx_);
  this->nbshape_ = std::vector<size_t>(shape_.begin() + bidx_, shape_.end());
  this->has_grad = has_grad; 
}

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape, size_t bidx, bool has_grad) {
  this->storage_ = data; 
  this->shape_ = shape;  
  this->bidx_ = bidx; 
  this->bshape_ = std::vector<size_t>(shape_.begin(), shape_.begin() + bidx_);
  this->nbshape_ = std::vector<size_t>(shape_.begin() + bidx_, shape_.end());
  this->has_grad = has_grad; 
}

Tensor::Tensor(std::vector<double> data, size_t bidx, bool has_grad) {
  this->storage_ = data; 
  std::vector<size_t> s = {data.size()};
  this->shape_ = s; 
  this->bidx_ = bidx; 
  this->bshape_ = std::vector<size_t>(shape_.begin(), shape_.begin() + bidx_);
  this->nbshape_ = std::vector<size_t>(shape_.begin() + bidx_, shape_.end());
  this->has_grad = has_grad; 
}

Tensor::Tensor(std::vector<std::vector<double>> data, size_t bidx, bool has_grad) {
  std::vector<size_t> shape = {data.size(), data[0].size()};
  CIntegrity::array_matches_shape(data, shape); 
  this->shape_ = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    res.insert(res.end(), data[i].begin(), data[i].end()); 
  }
  this->storage_ = res;  
  this->bidx_ = bidx; 
  this->bshape_ = std::vector<size_t>(shape_.begin(), shape_.begin() + bidx_);
  this->nbshape_ = std::vector<size_t>(shape_.begin() + bidx_, shape_.end());
  this->has_grad = has_grad; 
}

Tensor::Tensor(std::vector<std::vector<std::vector<double>>> data, size_t bidx, bool has_grad) {
  std::vector<size_t> shape = {data.size(), data[0].size(), data[0][0].size()}; 
  CIntegrity::array_matches_shape(data, shape); 
  this->shape_ = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      res.insert(res.end(), data[i][j].begin(), data[i][j].end()); 
    }
  }
  this->storage_ = res;  
  this->bidx_ = bidx; 
  this->bshape_ = std::vector<size_t>(shape_.begin(), shape_.begin() + bidx_);
  this->nbshape_ = std::vector<size_t>(shape_.begin() + bidx_, shape_.end());
  this->has_grad = has_grad; 
}

Tensor* Tensor::arange(int start, int stop, int step, bool has_grad) {
  std::vector<double> storage_ = {}; 
  for (int i = start; i < stop; i += step) {
    storage_.push_back(i);
  }
  return new Tensor(storage_, std::vector<size_t>{storage_.size()}, 0, has_grad);
}

Tensor* Tensor::linspace(double start, double stop, int numsteps, bool has_grad){
  std::vector<double> storage_ = {}; 
  double stepsize = (stop - start) / (numsteps - 1); 
  for (double i = start; i <= stop; i += stepsize) {
    storage_.push_back(i);
  }
  return new Tensor(storage_, std::vector<size_t>{storage_.size()}, 0, has_grad);
}

Tensor* Tensor::gaussian(std::vector<size_t> shape, double mean, double stddev, size_t bidx, bool has_grad) {
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

  return new Tensor(result, shape, bidx, has_grad);
}

Tensor* Tensor::uniform(std::vector<size_t> shape, double min, double max, size_t bidx, bool has_grad) {
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

  return new Tensor(result, shape, bidx, has_grad);
}

Tensor* Tensor::ones(std::vector<size_t> shape, size_t bidx, bool has_grad) {
  return new Tensor(std::vector<double> (CIntegrity::prod(shape), 1.0), shape, bidx, has_grad); 
}

Tensor* Tensor::zeros(std::vector<size_t> shape, size_t bidx, bool has_grad) {
  return new Tensor(std::vector<double>(CIntegrity::prod(shape), 0.0), shape, bidx, has_grad); 
}

