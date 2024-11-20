#include <vector>
#include "Tensor.h"

const double& BaseTensor::at(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::out_of_range("Number of indices does not match tensor dimensions.");
  }
  size_t flat_index = 0;
  size_t multiplier = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds.");
    }
    flat_index += indices[i] * multiplier;
    multiplier *= shape_[i];
  }
  return storage_[flat_index];
}

double& BaseTensor::at(const std::vector<size_t>& indices) {
  return const_cast<double&>(static_cast<const Tensor*>(this)->at(indices));
}

