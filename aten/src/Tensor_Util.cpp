#include "Tensor.h" 

const bool BaseTensor::operator==(BaseTensor& other) const {
  // First, compare shapes
  if (this->shape_ != other.shape_) {
    return false;
  }

  // Compare each element, considering floating-point precision
  const double epsilon = std::numeric_limits<double>::epsilon();
  for (size_t i = 0; i < this->data().size(); ++i) {
    if (std::abs(this->data()[i] - other.data()[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

const bool BaseTensor::operator!=(BaseTensor& other) const {
    return !(*this == other);
}

Tensor Tensor::copy() {
  return Tensor(this->storage_, this->shape_);
}
