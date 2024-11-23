#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(UtilsTest, Equality) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t2 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t3 = GradTensor({1., 2., 3., 4.}, {2, 2}, 0); 
  GradTensor t4 = GradTensor({1., 2., 3., 4., 5., 6.}, {3, 2}, 1); 
  GradTensor t5 = GradTensor({1., 2., 3., 5.}, {2, 2}, 1); 

  ASSERT_TRUE(t1 == t2);
  ASSERT_TRUE(t1 != t3);
  ASSERT_TRUE(t1 != t4);
  ASSERT_TRUE(t1 != t5);
}

TEST(UtilsTest, Copy) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
  GradTensor t2 = t1.copy(); 

  // They are copies
  ASSERT_TRUE(t1 == t2); 
  // But do not live in the same memory address 
  ASSERT_TRUE(&t1 != &t2);
}

