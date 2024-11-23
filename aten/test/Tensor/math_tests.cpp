#include <gtest/gtest.h> 
#include "Tensor.h"
#include <vector> 

TEST(TensorTest, Sum2) {
  Tensor t = Tensor(
    {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}, 
    {5, 2}
  ); 
  Tensor sum = t.sum();

  EXPECT_EQ(sum.data(), std::vector<double>{55.}); 
  sum.backprop(false); 

  GradTensor grad_gt = GradTensor(
    std::vector<double>(10, 1.), 
    {1, 5, 2}, 
    1
  );

  ASSERT_TRUE(grad_gt == t.grad); 

}




