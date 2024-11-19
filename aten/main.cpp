#include "./src/Tensor.h"
#include <vector>

int main() {
  std::vector<double> data = std::vector<double>(2.0);
  std::vector<size_t> shape = std::vector<size_t>(1);
  Tensor t = Tensor(data, shape); 
}
