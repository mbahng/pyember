#include <vector> 
#include <ctime>
#include "../Tensor.h"

void array_matches_shape(
  std::vector<double> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
);

ScalarTensor::ScalarTensor() 
  : Tensor({0.}, {1}){}

ScalarTensor::ScalarTensor(double data) 
  : Tensor({data}, {1}) {}

ScalarTensor::ScalarTensor(std::vector<double> data) 
  : Tensor(data, {1}) {
    if (data.size() != 1) {
      throw std::logic_error("The data is not of length 1.");
    }
  }


