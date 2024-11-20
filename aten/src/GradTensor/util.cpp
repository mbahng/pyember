#include "../Tensor.h"

bool GradTensor::operator==(GradTensor& other) const {
  return (BaseTensor::operator==(other) && this->pivot_ == other.pivot_);
}

bool GradTensor::operator!=(GradTensor& other) const {
    return !(*this == other);
}
GradTensor GradTensor::copy() const {
  return GradTensor(this->storage_, this->shape_, this->pivot_);
}


