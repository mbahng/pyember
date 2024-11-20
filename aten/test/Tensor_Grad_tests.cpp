#include <gtest/gtest.h> 
#include "Tensor.h"
#include <vector>

TEST(GradTensorTest, Matmul) {
  std::vector<double> storage1 = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}; 
  std::vector<size_t> shape1 = {5, 2}; 
  GradTensor t1 = GradTensor(storage1, shape1, 1); 

  std::vector<double> storage2= {2., 3., 4., 5., 7., 8., 9., 10}; 
  std::vector<size_t> shape2 = {2, 4}; 
  GradTensor t2 = GradTensor(storage2, shape2, 1); 
  
  GradTensor prod = t1.rmatmul(t2); 

  std::cout << std::string(t1); 
  std::cout << std::string(t2); 
  std::cout << std::string(prod); 
}

