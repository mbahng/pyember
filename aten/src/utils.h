#pragma once
#include <vector>
#include <iostream>

int shape_to_length(std::vector<size_t> shape);

void array_matches_shape(std::vector<double> data, std::vector<size_t> shape);

void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
  );

void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
  );

std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape);

template <typename T>
void print(const std::vector<T>& v) {
  std::cout << "( ";
  for (T p : v) {
    std::cout << p << " ";
  }
  std::cout << ")\n";
}

template void print<double>(const std::vector<double>&); 
template void print<size_t>(const std::vector<size_t>&); 

template <typename T>
std::vector<T> concat(const std::vector<T>& v) {
  return v;  // Base case: single vector
}

template <typename T, typename... Args>
std::vector<T> concat(const std::vector<T>& v1, const std::vector<T>& v2, Args... args) {
  std::vector<T> result = v1;
  result.insert(result.end(), v2.begin(), v2.end());
  return concat(result, args...);  // Recursive call with remaining vectors
} 

template std::vector<double> concat(const std::vector<double>&, const std::vector<double>&);
template std::vector<size_t> concat(const std::vector<size_t>&, const std::vector<size_t>&);

template <typename T> 
std::vector<T> duplicate(const std::vector<T> v) {
  return concat(v, v); 
}

bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape);

template std::vector<double> duplicate(const std::vector<double>);
template std::vector<size_t> duplicate(const std::vector<size_t>);

std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx);

size_t prod(std::vector<size_t> input);
std::vector<double> range(int l, int u, int s);
std::vector<double> range(int u, int s = 1);

