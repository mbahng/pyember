#include "./src/Tensor.h"
#include <vector>

int main() {
  GradTensor g1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 0);
  GradTensor g2 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
  std::cout << (bool)(g1 == g1); 
}
