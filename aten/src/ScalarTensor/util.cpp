#include "../Tensor.h"

ScalarTensor ScalarTensor::copy() const {
  return ScalarTensor(this->storage_);
}

double ScalarTensor::item() const {
  return storage_[0];
}

