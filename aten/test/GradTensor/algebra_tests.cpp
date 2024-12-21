#include <vector>
#include <gtest/gtest.h>
#include "../../src/Tensor.h"

std::vector<double> range(int l, int u, int s);
std::vector<double> range(int u, int s = 1); 
std::vector<double> concat(std::vector<double> v1, std::vector<double> v2); 

void print(std::vector<double> input);

namespace GT_Add_GT {

  TEST(GradTensorTest, NB_add_NB) {
    GradTensor* t1 = new GradTensor(range(1, 7, 1), {2, 3}, 1, 0);
    GradTensor* t2 = new GradTensor(range(1, 13, 1), {2, 2, 3}, 2, 1); 
    GradTensor truth = GradTensor(concat(range(2, 14, 2), range(8, 20, 2)), {2, 2, 3}, 2, 1); 
    GradTensor* s1 = t1->add(t2); 
    ASSERT_TRUE(*s1 == truth); 
    delete t1; 
    delete t2; 
    delete s1;
  }

  TEST(GradTensorTest, B_add_NB) {
    GradTensor* t1 = new GradTensor(range(1, 13, 1), {2, 2, 3}, 2, 1); 
    GradTensor* t2 = new GradTensor(range(1, 7, 1), {2, 3}, 1, 0);
    GradTensor truth = GradTensor(concat(range(2, 14, 2), range(8, 20, 2)), {2, 2, 3}, 2, 1);
    GradTensor* s1 = t1->add(t2); 
    ASSERT_TRUE(*s1 == truth);
    delete t1; 
    delete t2; 
    delete s1;
  }

  TEST(GradTensorTest, NB_add_B) {
    GradTensor* t1 = new GradTensor(range(1, 7, 1), {2, 3}, 1, 0);
    GradTensor* t2 = new GradTensor(range(1, 13, 1), {2, 2, 3}, 2, 1); 
    GradTensor truth = GradTensor(concat(range(2, 14, 2), range(8, 20, 2)), {2, 2, 3}, 2, 1);
    GradTensor* s1 = t1->add(t2); 
    ASSERT_TRUE(*s1 == truth);
    delete t1; 
    delete t2; 
    delete s1;
  }

  TEST(GradTensorTest, B_add_B) {
    GradTensor* t1 = new GradTensor(range(1, 13, 1), {2, 2, 3}, 2, 1); 
    GradTensor* t2 = new GradTensor(range(1, 13, 1), {2, 2, 3}, 2, 1); 
    std::vector<double> data1 = concat(range(2, 14, 2), range(8, 20, 2)); 
    std::vector<double> data2 = concat(range(8, 20, 2), range(14, 26, 2)); 
    GradTensor truth = GradTensor(concat(data1, data2), {2, 2, 2, 3}, 3, 2);
    GradTensor* s1 = t1->add(t2); 
    ASSERT_TRUE(*s1 == truth);
    delete t1; 
    delete t2; 
    delete s1;
  }

}

namespace GT_Sub_GT { 

  TEST(GradTensorTest, SubGradTensor) {
    // regular adding GT + GT
    GradTensor* t1 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    GradTensor* t2 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    GradTensor* truth = new GradTensor({0., 0., 0., 0.}, {2, 2}, 1);
    GradTensor* s1 = t1->sub(t2); 
    ASSERT_TRUE(*s1 == *truth);
    // Should throw errors when shapes or pivots do not match 
    GradTensor* t3 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 2); 
    GradTensor* t4 = new GradTensor({1., 2., 3., 4., 5., 6.}, {3, 2}, 2); 
    ASSERT_THROW(t1->sub(t3), std::logic_error); 
    ASSERT_THROW(t1->sub(t4), std::logic_error);

    // Cleanup
    delete t1;
    delete t2;
    delete t3;
    delete t4;
    delete truth;
    delete s1;
  }

}

namespace GT_Mul_GT { 

  TEST(GradTensorTest, MulGradTensor) {
    // regular adding GT + GT
    GradTensor* t1 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    GradTensor* t2 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    GradTensor* truth = new GradTensor({1., 4., 9., 16.}, {2, 2}, 1);
    GradTensor* s1 = t1->mul(t2); 
    ASSERT_TRUE(*s1 == *truth);
    // Should throw errors when shapes or pivots do not match 
    GradTensor* t3 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 2); 
    GradTensor* t4 = new GradTensor({1., 2., 3., 4., 5., 6.}, {3, 2}, 2); 
    ASSERT_THROW(t1->mul(t3), std::logic_error); 
    ASSERT_THROW(t1->mul(t4), std::logic_error);

    // Cleanup
    delete t1;
    delete t2;
    delete t3;
    delete t4;
    delete truth;
    delete s1;
  }

}

namespace GT_Add_T {

  TEST(GradTensorTest, AddTensor) {
    // regular adding GT + GT
    GradTensor* t1 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    Tensor* t2 = new Tensor({1., 2., 3., 4.}, {2, 2});
    Tensor* truth = new Tensor({2., 4., 6., 8.}, {2, 2});
    Tensor* s1 = t1->add(t2); 
    ASSERT_TRUE(*s1 == *truth);
    // Should throw errors when shapes or pivots do not match 
    Tensor* t3 = new Tensor({1., 2., 3., 4.}, {3, 2}); 
    ASSERT_THROW(t1->add(t3), std::logic_error); 
  }
  
}

namespace GT_Sub_T {

  TEST(GradTensorTest, SubTensor) {
    // regular adding GT + GT
    GradTensor* t1 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    Tensor* t2 = new Tensor({1., 2., 3., 4.}, {2, 2});
    Tensor* truth = new Tensor({0., 0., 0., 0.}, {2, 2});
    Tensor* s1 = t1->sub(t2); 
    ASSERT_TRUE(*s1 == *truth);
    // Should throw errors when shapes or pivots do not match 
    Tensor* t3 = new Tensor({1., 2., 3., 4.}, {3, 2}); 
    ASSERT_THROW(t1->sub(t3), std::logic_error);

    // Cleanup
    delete t1;
    delete t2;
    delete t3;
    delete truth;
    delete s1;
  }
  
}

namespace GT_Mul_T { 

  TEST(GradTensorTest, MulTensor) {
    // regular adding GT + GT
    GradTensor* t1 = new GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
    Tensor* t2 = new Tensor({1., 2., 3., 4.}, {2, 2}); 
    Tensor* truth = new Tensor({1., 4., 9., 16.}, {2, 2});
    Tensor* s1 = t1->mul(t2); 
    ASSERT_TRUE(*s1 == *truth);
    // Should throw errors when shapes or pivots do not match 
    Tensor* t3 = new Tensor({1., 2., 3., 4.}, {3, 2}); 
    ASSERT_THROW(t1->mul(t3), std::logic_error);

    // Cleanup
    delete t1;
    delete t2;
    delete t3;
    delete truth;
    delete s1;
  }

}

namespace GT_Mul_GT {

  TEST(GradTensorTest, MatMulGradTensor) {
    GradTensor* t1 = new GradTensor(
        {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}, 
        {5, 2}, 1); 
    GradTensor* t2 = new GradTensor(
        {2., 3., 4., 5., 7., 8., 9., 10}, 
        {2, 4}, 1); 
    GradTensor* gt = new GradTensor(
      {
        16., 19., 22., 25., 34., 41., 48., 55., 52., 63., 
        74., 85., 70., 85., 100., 115., 88., 107., 126., 145.
      }, 
      {5, 4}, 1);
    GradTensor* result = t1->matmul(t2);
    ASSERT_TRUE(*result == *gt);

    // Cleanup
    delete t1;
    delete t2;
    delete gt;
    delete result;
  }

}

