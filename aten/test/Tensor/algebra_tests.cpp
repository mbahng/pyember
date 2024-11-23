#include <gtest/gtest.h> 
#include "../../src/Tensor.h"
#include <vector>

TEST(TensorAlgebraTest, AddTensorVector) {
  Tensor t1 = Tensor::arange(0, 5);
  Tensor t2 = Tensor::arange(0, 5); 
  Tensor s1 = t1.add(t2);
  Tensor truth_sum = Tensor::arange(0, 10, 2);
  
  ASSERT_TRUE(s1 == truth_sum); 
  
  s1.backprop(true);

  GradTensor truth = GradTensor::eye(5, 1);
 
  ASSERT_TRUE(t1.grad == truth); 
  ASSERT_TRUE(t2.grad == truth); 
}

TEST(TensorAlgebraTest, AddGradTensorVector) {

}

TEST(TensorAlgebraTest, SubTensor) {
  Tensor t1 = Tensor::arange(0, 5);
  Tensor t2 = Tensor::arange(0, 5); 
  Tensor s1 = t1.sub(t2);
  Tensor truth_sum = Tensor::zeros({5});
  
  ASSERT_TRUE(s1 == truth_sum); 
  
  s1.backprop(true);

  GradTensor truth1 = GradTensor::eye(5, 1);
  GradTensor truth2 = GradTensor::eye(5, 1); 

  for (int i = 0; i < truth2.storage_.size(); i++) {
    truth2.storage_[i] *= -1;
  }

  ASSERT_TRUE(t1.grad == truth1); 
  ASSERT_TRUE(t2.grad == truth2); 
}

TEST(TensorAlgebraTest, SubGradTensor) {

}

TEST(TensorAlgebraTest, MulTensor) {

}

TEST(TensorAlgebraTest, MulGradTensor) {

}

TEST(TensorAlgebraTest, MatMulTensor) {

}


