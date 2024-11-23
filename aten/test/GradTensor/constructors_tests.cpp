#include <vector>
#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(UtilsTst, Matmul) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 

  std::vector<double> data = {1., 2., 3., 4.}; 
}
