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

