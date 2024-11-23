#include <vector>
#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(GradTensorTest, GradSame) {
  std::vector<double> storage1 = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}; 
  std::vector<size_t> shape1 = {5, 2}; 
  GradTensor t1 = GradTensor(storage1, shape1, 1); 

  std::vector<double> storage2= {2., 3., 4., 5., 7., 8., 9., 10}; 
  std::vector<size_t> shape2 = {2, 4}; 
  GradTensor t2 = GradTensor(storage2, shape2, 1); 
  
  GradTensor prod = t1.matmul(t2); 
  GradTensor gt = GradTensor(
    {
      16., 19., 22., 25., 34., 41., 48., 55., 52., 63., 
      74., 85., 70., 85., 100., 115., 88., 107., 126., 145.
    }, 
    std::vector<size_t>{5, 4}, 
    1
  );
  EXPECT_EQ(prod.data(), gt.data());
}



