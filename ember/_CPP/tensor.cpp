#include "tensor.h"

int shape_to_length(std::vector<int> shape) {
  int len = 1; 
  for (int i = 0; i < shape.size(); ++i) {
    len *= shape[i]; 
  }
  return len; 
}

