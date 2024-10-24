#include "tensor.h"
#include <random>
#include <chrono>
#include <vector>
#include <atomic>

int shape_to_length(std::vector<int> shape) {
  int len = 1; 
  for (int i = 0; i < shape.size(); ++i) {
    len *= shape[i]; 
  }
  return len; 
}


std::vector<std::vector<double>> zero_matrix(int n, int m) {
  std::vector<std::vector<double>> out(n, std::vector<double>(m, 0.0)); 
  return out; 
}

std::vector<std::vector<double>> zero_matrix(int n) {
  return zero_matrix(n, n); 
}

std::vector<std::vector<double>> eye_matrix(int n) {
  std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.0)); 
  for (int i = 0; i < n; i++) {
    out[i][i] = 1.; 
  }
  return out; 
}


/* static std::atomic<unsigned long long> seed_counter{0}; */
/*  */
/* Tensor gaussian(std::vector<int> shape, double mean, double stddev) { */
/*   // Create a unique seed by combining high-resolution time and a counter */
/*   auto now = std::chrono::high_resolution_clock::now(); */
/*   auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(); */
/*   unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32); */
/*  */
/*   // Create a generator with the unique seed */
/*   std::mt19937 generator(unique_seed); */
/*  */
/*   // Create a distribution */
/*   std::normal_distribution<double> distribution(mean, stddev); */
/*  */
/*   // Calculate total number of elements */
/*   int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()); */
/*  */
/*   // Create and fill the vector */
/*   std::vector<double> result(length); */
/*   for (int i = 0; i < length; ++i) { */
/*     result[i] = distribution(generator); */
/*   } */
/*  */
/*   return Tensor(result, shape, std::vector<Tensor>()); */
/* } */
/*  */
/* Tensor uniform(std::vector<int> shape, double min, double max) { */
/*   // (Use the same unique seeding method as in the gaussian function) */
/*   auto now = std::chrono::high_resolution_clock::now(); */
/*   auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(); */
/*   unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32); */
/*  */
/*   std::mt19937 generator(unique_seed); */
/*   std::uniform_real_distribution<double> distribution(min, max); */
/*  */
/*   int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()); */
/*   std::vector<double> result(length); */
/*   for (int i = 0; i < length; ++i) { */
/*     result[i] = distribution(generator); */
/*   } */
/*  */
/*   return Tensor(result, shape, std::vector<Tensor>()); */
/* } */
/*  */
/* Tensor ones(std::vector<int> shape) { */
/*   std::vector<double> data(shape_to_length(shape), 1.0);  */
/*   return Tensor(data, shape, std::vector<Tensor>());  */
/* } */
/*  */
/* Tensor zeros(std::vector<int> shape) { */
/*   std::vector<double> data(shape_to_length(shape), 0.0);  */
/*   return Tensor(data, shape, std::vector<Tensor>());  */
/* } */
