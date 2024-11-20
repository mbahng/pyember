// Initializers for Tensors 
#include <vector> 
#include <random> 
#include <ctime>
#include "Tensor.h"

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

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape) {
  this->storage_ = data; 
  this->shape_ = shape;  
}

Tensor::Tensor(std::vector<double> data) {
  this->storage_ = data; 
  std::vector<size_t> s = {data.size()};
  this->shape_ = s; 
}

Tensor::Tensor(std::vector<std::vector<double>> data) {
  std::vector<size_t> shape = {data.size(), data[0].size()};
  array_matches_shape(data, shape); 
  this->shape_ = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    res.insert(res.end(), data[i].begin(), data[i].end()); 
  }
  this->storage_ = res;  
}

Tensor::Tensor(std::vector<std::vector<std::vector<double>>> data) {
  std::vector<size_t> shape = {data.size(), data[0].size(), data[0][0].size()};
  array_matches_shape(data, shape); 
  this->shape_ = shape; 
  std::vector<double> res = {}; 
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      res.insert(res.end(), data[i][j].begin(), data[i][j].end()); 
    }
  }
  this->storage_ = res;  
}

Tensor Tensor::arange(int start, int stop, int step) {
  std::vector<double> storage_ = {}; 
  for (int i = start; i < stop; i += step) {
    storage_.push_back(i);
  }
  return Tensor(storage_, std::vector<size_t>{storage_.size()});
}

Tensor Tensor::linspace(double start, double stop, int numsteps){
  std::vector<double> storage_ = {}; 
  double stepsize = (stop - start) / numsteps; 
  for (double i = start; i <= stop; i += stepsize) {
    storage_.push_back(i);
  }
  return Tensor(storage_, std::vector<size_t>{storage_.size()});
}

Tensor Tensor::gaussian(std::vector<size_t> shape, double mean, double stddev) {
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

  return Tensor(result, shape);
}

Tensor Tensor::uniform(std::vector<size_t> shape, double min, double max) {
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

  return Tensor(result, shape);
}

Tensor Tensor::ones(std::vector<size_t> shape) {
  std::vector<double> data(shape_to_length(shape), 1.0); 
  return Tensor(data, shape); 
}

Tensor Tensor::zeros(std::vector<size_t> shape) {
  std::vector<double> data(shape_to_length(shape), 0.0); 
  return Tensor(data, shape); 
}
