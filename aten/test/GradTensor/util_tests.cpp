#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(UtilsTst, Stuff1) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t2 = GradTensor({1., 2., 3., 4.}, {2, 2}, 0); 
 
  EXPECT_EQ(t1.data(), t2.data()); 
  EXPECT_EQ(t1.shape(), t2.shape()); 
  EXPECT_NE(t1.pivot(), t2.pivot());
}

