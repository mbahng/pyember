#include "./src/Tensor.h"
#include <vector>

int main() {
  std::vector<size_t> nshape = {10, 10};
  Tensor a = Tensor::gaussian(nshape, 0.0, 1.0);
  Tensor b = Tensor::gaussian(nshape, 0.0, 1.0);
  Tensor c = Tensor::gaussian(nshape, 0.0, 1.0);
  Tensor d = Tensor::gaussian(nshape, 0.0, 1.0);
  std::cout << std::string(d) << std::endl;  
}
