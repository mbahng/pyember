#include <gtest/gtest.h> 
#include "../../src/Tensor.h"
#include <vector> 

TEST(TensorTest, SumVector) {
  Tensor t = Tensor::arange(0, 10);
  Tensor sum = t.sum(); 
  Tensor truth = Tensor({45.});

  ASSERT_TRUE(sum == truth);

  sum.backprop(false); 

  GradTensor grad_gt = GradTensor(
    std::vector<double>(10, 1.), 
    {1, 10}, 
    1
  );

  ASSERT_TRUE(grad_gt == t.grad); 
}

TEST(TensorTest, SumMatrix) {
  Tensor t = Tensor::arange(0, 10).reshape({5, 2}); 
  Tensor sum = t.sum(); 
  Tensor truth = Tensor({45.});

  ASSERT_TRUE(sum == truth);

  sum.backprop(false); 

  GradTensor grad_gt = GradTensor(
    std::vector<double>(10, 1.), 
    {1, 5, 2}, 1);

  ASSERT_TRUE(grad_gt == t.grad); 
} 

TEST(TensorTest, SumTensor) {
  Tensor t = Tensor::arange(0, 24).reshape({2, 3, 4}); 
  Tensor sum = t.sum(); 
  Tensor truth = Tensor({276.});

  ASSERT_TRUE(sum == truth);

  sum.backprop(false); 

  GradTensor grad_gt = GradTensor(std::vector<double>(24, 1.), 
    {1, 2, 3, 4}, 1);

  ASSERT_TRUE(grad_gt == t.grad); 
}

