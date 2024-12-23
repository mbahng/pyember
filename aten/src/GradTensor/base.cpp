#include "../Tensor.h"
#include "../utils.h"

GradTensor::operator std::string() const {
  std::string result = BaseTensor::operator std::string();  
  if (result.back() != '\n') {
    result += ", pidx = " + std::to_string(this->pidx()) + '\n'; 
  }
  else {
    result.pop_back();
    result += ", pidx = " + std::to_string(this->pidx()) + '\n'; 
  }
  return result; 
}

bool GradTensor::operator==(GradTensor& other) const {
  return (BaseTensor::operator==(other) && this->pidx() == other.pidx());
}

bool GradTensor::operator!=(GradTensor& other) const {
    return !(*this == other);
}
GradTensor* GradTensor::copy() const {
  return new GradTensor(this->storage_, this->shape_, this->bidx_, this->pidx());
}

GradTensor* GradTensor::reshape(std::vector<size_t> new_shape, bool inplace) {
  if (prod(new_shape) != prod(this->shape_)) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  if (inplace) {
    this->shape_ = new_shape; 
    return this; 
  }
  else { 
    // usually pidx will become meaningless when you reshape it. 
    GradTensor* out = new GradTensor(storage_, new_shape, bidx_, pidx_);
    return out; 
  }
}

GradTensor& GradTensor::transpose(const std::vector<size_t>& axes) {
  // If no axes specified, reverse all dimensions
  std::vector<size_t> perm_axes = axes;
  if (perm_axes.empty()) {
    perm_axes.resize(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
      perm_axes[i] = shape_.size() - 1 - i;
    }
  }
  
  // Validate axes
  if (perm_axes.size() != shape_.size()) {
    throw std::invalid_argument("Number of axes must match tensor dimensions");
  }
  
  // Check for duplicates and valid range
  std::vector<bool> used(shape_.size(), false);
  for (size_t axis : perm_axes) {
    if (axis >= shape_.size()) {
      throw std::out_of_range("Axis index out of range");
  }
    if (used[axis]) {
      throw std::invalid_argument("Duplicate axis in permutation");
    }
    used[axis] = true;
  }
  
  // Calculate new shape
  std::vector<size_t> new_shape(shape_.size());
  for (size_t i = 0; i < shape_.size(); ++i) {
    new_shape[i] = shape_[perm_axes[i]];
  }
  
  // Create temporary storage for the transposed data
  std::vector<double> temp_storage(storage_.size());
  
  // Helper function to convert flat index to multidimensional indices
  auto flat_to_indices = [](size_t flat_idx, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      indices[i] = flat_idx % shape[i];
      flat_idx /= shape[i];
    }
    return indices;
  };
  
  // Helper function to convert multidimensional indices to flat index
  auto indices_to_flat = [](const std::vector<size_t>& indices, const std::vector<size_t>& shape) {
    size_t flat_idx = 0;
    size_t multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      flat_idx += indices[i] * multiplier;
      multiplier *= shape[i];
    }
    return flat_idx;
  };
  
  // Perform the transpose
  for (size_t i = 0; i < storage_.size(); ++i) {
      // Get original indices
      std::vector<size_t> orig_indices = flat_to_indices(i, shape_);
      
      // Create new indices based on permutation
      std::vector<size_t> new_indices(shape_.size());
      for (size_t j = 0; j < shape_.size(); ++j) {
          new_indices[j] = orig_indices[perm_axes[j]];
      }
      
      // Calculate new flat index and copy data to temporary storage
      size_t new_flat_idx = indices_to_flat(new_indices, new_shape);
      temp_storage[new_flat_idx] = storage_[i];
  }
  
  // Update shape and storage
  shape_ = new_shape;
  storage_ = std::move(temp_storage);
  
  return *this;
}

