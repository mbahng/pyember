#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(STAddT, Scalar) {
  ScalarTensor* c = new ScalarTensor(3.); 
  ScalarTensor* A = new ScalarTensor(2.); 
  ScalarTensor* B = new ScalarTensor(2.);  
  ScalarTensor* A2 = c->add(A); 
  ScalarTensor* B2 = c->add(B); 
  ScalarTensor* truth = new ScalarTensor(5.); 
  ASSERT_TRUE(*A2 == *truth);
  ASSERT_TRUE(*B2 == *truth);

  delete c;
  delete A;
  delete B;
  delete A2;
  delete B2;
  delete truth;
}

TEST(STaddT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 7, 1);
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STaddT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 7, 1)->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STaddT, Tensor) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 24, 1)->reshape({2, 3, 4});
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 27, 1)->reshape({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubT, Scalar) {
  ScalarTensor* c = new ScalarTensor(3.); 
  ScalarTensor* A = new ScalarTensor(2.); 
  ScalarTensor* B = new ScalarTensor(2);  
  ScalarTensor* A2 = c->sub(A); 
  ScalarTensor* B2 = c->sub(B); 
  ScalarTensor* truth = new ScalarTensor(1.); 
  ASSERT_TRUE(*A2 == *truth);
  ASSERT_TRUE(*B2 == *truth);

  delete c;
  delete A;
  delete B;
  delete A2;
  delete B2;
  delete truth;
}

TEST(STsubT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->sub(A); 
  Tensor* truth = new Tensor({3., 2., 1., 0.});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->sub(A); 
  Tensor* truth = (new Tensor({3., 2., 1., 0.}))->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubT, Tensor) {
  ScalarTensor* c = new ScalarTensor(2.); 
  Tensor* A = Tensor::ones({2, 3, 4});
  Tensor* A2 = c->sub(A); 
  Tensor* truth = Tensor::ones({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulT, Scalar) {
  ScalarTensor* c = new ScalarTensor(3.); 
  ScalarTensor* A = new ScalarTensor(2.); 
  ScalarTensor* B = new ScalarTensor(2);  
  ScalarTensor* A2 = c->mul(A); 
  ScalarTensor* B2 = c->mul(B); 
  ScalarTensor* truth = new ScalarTensor(6.); 
  ASSERT_TRUE(*A2 == *truth);
  ASSERT_TRUE(*B2 == *truth);

  delete c;
  delete A;
  delete B;
  delete A2;
  delete B2;
  delete truth;
}

TEST(STmulT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 12, 3);
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 12, 3)->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulT, Tensor) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 24, 1)->reshape({2, 3, 4});
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 72, 3)->reshape({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STaddGT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 7, 1);
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STaddGT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 7, 1)->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STaddGT, Tensor) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 24, 1)->reshape({2, 3, 4});
  Tensor* A2 = c->add(A); 
  Tensor* truth = Tensor::arange(3, 27, 1)->reshape({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubGT, Scalar) {
  ScalarTensor* c = new ScalarTensor(3.); 
  ScalarTensor* A = new ScalarTensor(2.); 
  Tensor* B = new Tensor({2});  
  Tensor* A2 = c->sub(A); 
  Tensor* B2 = c->sub(B); 
  Tensor* truth = new ScalarTensor(1.); 
  ASSERT_TRUE(*A2 == *truth);
  ASSERT_TRUE(*B2 == *truth);

  delete c;
  delete A;
  delete B;
  delete A2;
  delete B2;
  delete truth;
}

TEST(STsubGT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->sub(A); 
  Tensor* truth = new Tensor({3., 2., 1., 0.});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubGT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->sub(A); 
  Tensor* truth = (new Tensor({3., 2., 1., 0.}))->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STsubGT, Tensor) {
  ScalarTensor* c = new ScalarTensor(2.); 
  Tensor* A = Tensor::ones({2, 3, 4});
  Tensor* A2 = c->sub(A); 
  Tensor* truth = Tensor::ones({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulGT, Scalar) {
  ScalarTensor* c = new ScalarTensor(3.); 
  ScalarTensor* A = new ScalarTensor(2.); 
  ScalarTensor* B = new ScalarTensor(2);  
  ScalarTensor* A2 = c->mul(A); 
  ScalarTensor* B2 = c->mul(B); 
  ScalarTensor* truth = new ScalarTensor(6.); 
  ASSERT_TRUE(*A2 == *truth);
  ASSERT_TRUE(*B2 == *truth);

  delete c;
  delete A;
  delete B;
  delete A2;
  delete B2;
  delete truth;
}

TEST(STmulGT, Vector) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1);
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 12, 3);
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulGT, Matrix) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 4, 1)->reshape({2, 2});
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 12, 3)->reshape({2, 2});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}

TEST(STmulGT, Tensor) {
  ScalarTensor* c = new ScalarTensor(3.); 
  Tensor* A = Tensor::arange(0, 24, 1)->reshape({2, 3, 4});
  Tensor* A2 = c->mul(A); 
  Tensor* truth = Tensor::arange(0, 72, 3)->reshape({2, 3, 4});
  ASSERT_TRUE(*A2 == *truth);

  delete c;
  delete A;
  delete A2;
  delete truth;
}
