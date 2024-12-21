#include <vector> 
#include <ctime>
#include "../Tensor.h"

ScalarTensor::ScalarTensor() 
  : Tensor({0.}, std::vector<size_t>{1}){}

ScalarTensor::ScalarTensor(double data) 
  : Tensor({data}, std::vector<size_t>{1}) {}

ScalarTensor::ScalarTensor(std::vector<double> data) 
  : Tensor(data, std::vector<size_t>{1}) {
    if (data.size() != 1) {
      throw std::logic_error("The data is not of length 1.");
    }
  }


