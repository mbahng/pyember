#include "./src/Tensor.h"
#include <vector>

int main() {
  Tensor* t1 = Tensor::arange(0, 6)->reshape({2, 3}); 
  Tensor* t2 = t1->transpose();
  Tensor truth = Tensor({0., 3., 1., 4., 2., 5.}, {3, 2});

  std::cout << std::string(*t2) << std::endl;
  std::cout << std::string(truth) << std::endl;
  return 0;
}
