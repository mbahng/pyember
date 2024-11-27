#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(STAddT, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  ScalarTensor B = ScalarTensor(2.);  
  ScalarTensor A2 = c.add(A); 
  ScalarTensor B2 = c.add(B); 
  ScalarTensor truth = ScalarTensor(5.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(STaddT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1);
  ASSERT_TRUE(A2 == truth);
}

TEST(STaddT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STaddT, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 27, 1).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubT, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  ScalarTensor B = ScalarTensor(2);  
  ScalarTensor A2 = c.sub(A); 
  ScalarTensor B2 = c.sub(B); 
  ScalarTensor truth = ScalarTensor(1.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(STsubT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.}).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubT, Tensor) {
  ScalarTensor c = ScalarTensor(2.); 
  Tensor A = Tensor::ones({2, 3, 4});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor::ones({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulT, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  ScalarTensor B = ScalarTensor(2);  
  ScalarTensor A2 = c.mul(A); 
  ScalarTensor B2 = c.mul(B); 
  ScalarTensor truth = ScalarTensor(6.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(STmulT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3);
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulT, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 72, 3).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(STaddGT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1);
  ASSERT_TRUE(A2 == truth);
}

TEST(STaddGT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 7, 1).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STaddGT, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.add(A); 
  Tensor truth = Tensor::arange(3, 27, 1).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubGT, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  Tensor B = Tensor({2});  
  Tensor A2 = c.sub(A); 
  Tensor B2 = c.sub(B); 
  Tensor truth = ScalarTensor(1.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(STsubGT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubGT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor({3., 2., 1., 0.}).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STsubGT, Tensor) {
  ScalarTensor c = ScalarTensor(2.); 
  Tensor A = Tensor::ones({2, 3, 4});
  Tensor A2 = c.sub(A); 
  Tensor truth = Tensor::ones({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulGT, Scalar) {
  ScalarTensor c = ScalarTensor(3.); 
  ScalarTensor A = ScalarTensor(2.); 
  ScalarTensor B = ScalarTensor(2);  
  ScalarTensor A2 = c.mul(A); 
  ScalarTensor B2 = c.mul(B); 
  ScalarTensor truth = ScalarTensor(6.); 
  ASSERT_TRUE(A2 == truth);
  ASSERT_TRUE(B2 == truth);
}

TEST(STmulGT, Vector) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1);
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3);
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulGT, Matrix) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 4, 1).reshape({2, 2});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 12, 3).reshape({2, 2});
  ASSERT_TRUE(A2 == truth);
}

TEST(STmulGT, Tensor) {
  ScalarTensor c = ScalarTensor(3.); 
  Tensor A = Tensor::arange(0, 24, 1).reshape({2, 3, 4});
  Tensor A2 = c.mul(A); 
  Tensor truth = Tensor::arange(0, 72, 3).reshape({2, 3, 4});
  ASSERT_TRUE(A2 == truth);
}
