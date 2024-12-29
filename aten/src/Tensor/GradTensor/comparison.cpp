#include "../Tensor.h"

bool GradTensor::operator==(GradTensor& other) const {
  return (BaseTensor::operator==(other) && this->pidx() == other.pidx());
}

bool GradTensor::operator!=(GradTensor& other) const {
    return !(*this == other);
}

bool GradTensor::operator<(GradTensor& other) const {
  if (this->is_scalar() && other.is_scalar()) {
    return this->item() < other.item(); 
  }
  if (this->pidx() != other.pidx()) {
    throw std::logic_error("You cannot inequality comare between tensors of differing pivots.");
  }
  return (BaseTensor::operator<(other));
}

bool GradTensor::operator>(GradTensor& other) const { 
  if (this->is_scalar() && other.is_scalar()) {
    return this->item() > other.item(); 
  }
  if (this->pidx() != other.pidx()) {
    throw std::logic_error("You cannot inequality comare between tensors of differing pivots.");
  }
  return (BaseTensor::operator>(other));
}

bool GradTensor::operator<=(GradTensor& other) const {
  return *this < other || *this == other;
}

bool GradTensor::operator>=(GradTensor& other) const {
  return *this > other || *this == other;
}
