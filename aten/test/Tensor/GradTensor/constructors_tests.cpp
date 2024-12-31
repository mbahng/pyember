#include <gtest/gtest.h> 
#include <vector>
#include "../../../src/Tensor/Tensor.h"

TEST(UtilsTst, FullConstructor) {
  GradTensor* g = new GradTensor({1., 2., 3., 4.}, {2, 2}, 0, 1); 
  std::vector<double> g_data = {1., 2., 3., 4.}; 
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g->storage(), g_data); 
  ASSERT_EQ(g->shape(), g_shape); 
  ASSERT_EQ(g->pidx(), 1); 
}

TEST(UtilsTst, DefaultConstructor) {
  GradTensor* g = new GradTensor({2, 2}, 0, 1); 
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g->storage(), std::vector<double>(4, 0.)); 
  ASSERT_EQ(g->shape(), g_shape); 
  ASSERT_EQ(g->pidx(), 1); 
}

TEST(UtilsTst, EyeConstructor) {
  GradTensor* g = GradTensor::eye(2, 0, 1); 
  std::vector<double> g_data = {1., 0., 0., 1.};
  std::vector<size_t> g_shape = {2, 2}; 
  ASSERT_EQ(g->storage(), g_data);
  ASSERT_EQ(g->shape(), g_shape); 
  ASSERT_EQ(g->pidx(), 1); 
}

