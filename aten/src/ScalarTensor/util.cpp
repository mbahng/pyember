#include "../Tensor.h"

ScalarTensor* ScalarTensor::copy() const {
  return new ScalarTensor(this->storage_);
}

double ScalarTensor::item() const {
  return storage_[0];
}

