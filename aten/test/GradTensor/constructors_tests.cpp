#include <vector>
#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(UtilsTst, NullConstructor) {
  GradTensor g1 = GradTensor(); 
  GradTensor g2 = GradTensor(); 

  ASSERT_EQ(g1.data(), std::vector<double>{}); 
  ASSERT_EQ(g1.shape(), std::vector<size_t>{}); 
  ASSERT_EQ(g1.pivot(), 0); 
  ASSERT_TRUE(g1 == g2);
}

TEST(UtilsTst, FullConstructor) {
  GradTensor g = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  std::vector<double> g_data = {1., 2., 3., 4.}; 
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g.data(), g_data); 
  ASSERT_EQ(g.shape(), g_shape); 
  ASSERT_EQ(g.pivot(), 1); 
}

TEST(UtilsTst, DefaultConstructor) {
  GradTensor g = GradTensor({2, 2}, 1); 
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g.data(), std::vector<double>(4, 0.)); 
  ASSERT_EQ(g.shape(), g_shape); 
  ASSERT_EQ(g.pivot(), 1); 
}

TEST(UtilsTst, EyeConstructor) {
  GradTensor* g = GradTensor::eye(2, 1); 
  std::vector<double> g_data = {1., 0., 0., 1.};
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g->data(), g_data);
  ASSERT_EQ(g->shape(), g_shape); 
  ASSERT_EQ(g->pivot(), 1); 
}

