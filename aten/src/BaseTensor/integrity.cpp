#include <vector>
#include <cassert> 

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

std::vector<size_t> concat_indices(
    std::vector<size_t> shape1,
    std::vector<size_t> shape2) {
  std::vector<size_t> pairshape = shape1;
  pairshape.insert(pairshape.end(), shape2.begin(), shape2.end());
  return pairshape;
}

std::vector<size_t> duplicate_indices(
    std::vector<size_t> shape) {
  return concat_indices(shape, shape);
}

std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx) {
  std::vector<std::vector<size_t>> result(2);
  
  // First part: [0, idx)
  result[0].assign(shape.begin(), shape.begin() + idx);
  
  // Second part: [idx, end)
  result[1].assign(shape.begin() + idx, shape.end());
  return result; 
}

