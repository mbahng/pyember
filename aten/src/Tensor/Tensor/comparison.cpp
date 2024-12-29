#include "../Tensor.h"

bool Tensor::operator==(Tensor& other) const {
  return BaseTensor::operator==(other); 
}

bool Tensor::operator!=(Tensor& other) const {
    return !(*this == other);
}

bool Tensor::operator>(Tensor& other) const { 
  return (BaseTensor::operator>(other));
}

bool Tensor::operator<(Tensor& other) const {
  return (BaseTensor::operator<(other));
}

bool Tensor::operator>=(Tensor& other) const {
  return *this > other || *this == other;
}

bool Tensor::operator<=(Tensor& other) const {
  return *this < other || *this == other;
}


