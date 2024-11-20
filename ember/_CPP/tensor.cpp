#include "tensor.h"
#include <vector>

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

std::vector<std::vector<double>> eye_matrix(int n, double k) {
  std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.0)); 
  for (int i = 0; i < n; i++) {
    out[i][i] = k; 
  }
  return out; 
}
