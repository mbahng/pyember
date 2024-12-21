#include <vector>
#include <cassert> 
#include "utils.h"

int shape_to_length(std::vector<size_t> shape) { 
  int len = 1; 
  for (int i = 0; i < shape.size(); ++i) {
    len *= shape[i]; 
  }
  return len; 
}

void array_matches_shape(
  std::vector<double> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 1); 
  assert(data.size() == shape[0]);
}

void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 2); 
  assert(data.size() == shape[0]); 

  for (int i = 0; i < shape[0]; i++) {
    assert(data[i].size() == shape[1]); 
  }
}

void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 3); 
  assert(data.size() == shape[0]); 

  for (int i = 0; i < shape[0]; i++) {
    assert(data[i].size() == shape[1]); 
    for (int j = 0; j < shape[1]; j++) {
      assert(data[i][j].size() == shape[2]);
    }
  }
}

// Helper function to increment the indices
bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    indices[i]++;
    if (indices[i] < shape[i]) {
      return true;  // Successfully incremented
    }
    indices[i] = 0;  // Reset this position and continue with next position
  }
  return false;  // We've gone through all possibilities
}

// Function to generate all possible vectors
std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape) {
  std::vector<std::vector<size_t>> result;
  
  // Calculate total number of combinations
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }
  result.reserve(total);  // Reserve space for efficiency
  
  // Start with all zeros
  std::vector<size_t> current(shape.size(), 0);
  
  // Add first combination
  result.push_back(current);
  
  // Generate all other combinations
  while (increment_indices(current, shape)) {
    result.push_back(current);
  }
  
  return result;
}

std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx) {
  std::vector<std::vector<size_t>> result(2);
  
  // First part: [0, idx)
  result[0].assign(shape.begin(), shape.begin() + idx);
  
  // Second part: [idx, end)
  result[1].assign(shape.begin() + idx, shape.end());
  return result; 
}

size_t prod(std::vector<size_t> input) {
  int product = 1; 
  for (const auto& num : input) {
    product *= num; 
  }
  return product; 
}

std::vector<double> range(int l, int u, int s) {
  std::vector<double> res; 
  for (int p = l; p < u; p += s) {
    res.push_back(p);
  }
  return res; 
}

std::vector<double> range(int u, int s) {
  return range(0, u, s);
} 

