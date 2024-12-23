#pragma once
#include <vector>
#include <iostream>
#include "Tensor.h"

namespace CIntegrity  {
  // Constructor Integrity
  
  size_t prod(std::vector<size_t> input);

  void array_matches_shape(std::vector<double> data, std::vector<size_t> shape);

  void array_matches_shape(
    std::vector<std::vector<double>> data, 
    std::vector<size_t> shape
  );

  void array_matches_shape(
    std::vector<std::vector<std::vector<double>>> data, 
    std::vector<size_t> shape
  );

  std::vector<double> range(int l, int u, int s);
  std::vector<double> range(int u, int s = 1);
}

namespace OIntegrity {
  // Operation Integrity

  struct Shape {
    std::vector<size_t> shape; 
    std::vector<size_t> b_shape; 
    std::vector<size_t> nb_shape; 
    size_t bidx; 
    size_t pidx; 
  };

  Shape compat(Tensor* t1, Tensor* t2);
  Shape compat(GradTensor* t1, Tensor* t2);
  Shape compat(Tensor* t1, GradTensor* t2);
  Shape compat(GradTensor* t1, GradTensor* t2);
  Shape matmul_compat(GradTensor* t1, GradTensor* t2); 
  Shape matmul_compat(Tensor* t1, Tensor* t2); 
}

namespace Debug {

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

  void print(const Tensor* t);
  void print(const Tensor t);
  void print(const GradTensor* t);
  void print(const GradTensor t);
}

namespace Index {

  bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape);

  std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape);

  std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx);

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

  template std::vector<size_t> concat(const std::vector<size_t>&, const std::vector<size_t>&);
}
