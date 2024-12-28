#include "../Tensor.h"
#include <cxxabi.h>

bool BaseTensor::operator==(BaseTensor& other) const {
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


