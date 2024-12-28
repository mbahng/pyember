#include "../Tensor.h"

bool GradTensor::operator==(GradTensor& other) const {
  return (BaseTensor::operator==(other) && this->pidx() == other.pidx());
}

bool GradTensor::operator!=(GradTensor& other) const {
    return !(*this == other);
}


