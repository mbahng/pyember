#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(ScalarTensorAddTest, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  Tensor B = Tensor({2});  
  Tensor A2 = c.add(A); 
  Tensor B2 = c.add(B); 
  Tensor truth = ScalarTensor(5.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(ScalarTensorAddTest, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1);
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorAddTest, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorAddTest, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 27, 1).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorSubTest, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  Tensor B = Tensor({2});  
  Tensor A2 = c.sub(A); 
  Tensor B2 = c.sub(B); 
  Tensor truth = ScalarTensor(1.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(ScalarTensorSubTest, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorSubTest, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.}).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorSubTest, Tensor) {
  ScalarTensor c = ScalarTensor(2.); 
  Tensor A = Tensor::ones({2, 3, 4});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor::ones({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorMulTest, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  Tensor B = Tensor({2});  
  Tensor A2 = c.mul(A); 
  Tensor B2 = c.mul(B); 
  Tensor truth = ScalarTensor(6.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(ScalarTensorMulTest, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3);
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorMulTest, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(ScalarTensorMulTest, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 72, 3).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}
