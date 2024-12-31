#include "../Tensor.h"

bool BaseTensor::operator==(BaseTensor& other) const { 
  if (this->is_scalar() && other.is_scalar()) {
    return this->item() == other.item(); 
  }

  if (this->bshape() != other.bshape()) { return false; }
  if (this->nbshape() != other.nbshape()) { return false; }

  // Compare each element, considering floating-point precision
  const double epsilon = std::numeric_limits<double>::epsilon();
  for (size_t i = 0; i < this->storage().size(); ++i) {
    if (std::abs(this->storage()[i] - other.storage()[i]) > epsilon) { 
      return false;
    }
  }

  return true;
}

bool BaseTensor::operator!=(BaseTensor& other) const {
  return !(*this == other);
}

bool BaseTensor::operator>(BaseTensor& other) const{
  if (this->is_scalar() && other.is_scalar()) { 
    std::cout << this->item() << " > " << other.item() << "\n"; 
    return this->item() > other.item(); 
  }

  if (this->shape() != other.shape()) {
    throw std::logic_error("You cannot inequality compare between tensors of differing shapes. ");
  } 
  for (size_t i = 0; i < this->storage().size(); ++i) {
    if (this->storage()[i] <= other.storage()[i]) {
      return false;
    }
  }
  return true; 
}

bool BaseTensor::operator<(BaseTensor& other) const{
  if (this->is_scalar() && other.is_scalar()) {
    std::cout << this->item() << " < " << other.item() << "\n"; 
    return this->item() < other.item(); 
  }

  if (this->shape() != other.shape()) {
    throw std::logic_error("You cannot inequality compare between tensors of differing shapes. ");
  } 
  for (size_t i = 0; i < this->storage().size(); ++i) {
    if (this->storage()[i] >= other.storage()[i]) {
      return false;
    }
  }
  return true; 
}

bool BaseTensor::operator>=(BaseTensor& other) const{
  return *this > other || *this == other;
}

bool BaseTensor::operator<=(BaseTensor& other) const{
  return *this < other || *this == other;
}

