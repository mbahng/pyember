#include <vector>
#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(ScalarTensorUtilsTst, NullConstructor) {
  ScalarTensor s1 = ScalarTensor();
  ScalarTensor s2 = ScalarTensor(); 

  ASSERT_EQ(s1.data(), std::vector<double>{0.}); 
  ASSERT_EQ(s1.shape(), std::vector<size_t>{}); 
  ASSERT_TRUE(s1 == s2);
}

TEST(ScalarTensorUtilsTst, ScalarConstructor) {
  ScalarTensor s1 = ScalarTensor(2.);
  ScalarTensor s2 = ScalarTensor(2.); 

  ASSERT_EQ(s1.data(), std::vector<double>{2.}); 
  ASSERT_EQ(s1.shape(), std::vector<size_t>{}); 
  ASSERT_TRUE(s1 == s2);
}

TEST(ScalarTensorUtilsTst, VectorConstructor) {
  std::vector<double> s_data = {2.}; 
  ScalarTensor s1 = ScalarTensor(s_data);
  ScalarTensor s2 = ScalarTensor(s_data); 

  ASSERT_EQ(s1.data(), std::vector<double>{2.}); 
  ASSERT_EQ(s1.shape(), std::vector<size_t>{}); 
  ASSERT_TRUE(s1 == s2);
}
