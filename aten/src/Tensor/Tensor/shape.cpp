#include "../Tensor.h"

Tensor* Tensor::shallowcopy(bool requires_grad) const {
  // creates a shallow copy 
  return new Tensor(this->_storage, this->_shape, this->bidx, requires_grad); 
}

Tensor* Tensor::deepcopy(bool requires_grad) const {
  // creates a deep copy 
  std::vector<double> storage = this->_storage; 
  std::vector<size_t> shape = this->_shape;  
  size_t bidx = this->bidx; 
  return new Tensor(storage, shape, bidx, requires_grad); 
}

Tensor* Tensor::copy(bool requires_grad) const {
  // alias for shallow copy
  return this->shallowcopy(requires_grad);
}

Tensor* Tensor::reshape(std::vector<size_t> new_shape, bool inplace, bool requires_grad) { 
  if (inplace) {
    this->_shape = new_shape; 
    return this; 
  }
  else {
    Tensor* out = new Tensor(_storage, new_shape, requires_grad);
    return out; 
  }
}

Tensor* Tensor::transpose(const std::vector<size_t>& axes, bool inplace, bool requires_grad) {
  // If no axes specified, reverse all dimensions
  std::vector<size_t> perm_axes = axes;
  if (perm_axes.empty()) {
      perm_axes.resize(_shape.size());
      for (size_t i = 0; i < _shape.size(); ++i) {
          perm_axes[i] = _shape.size() - 1 - i;
      }
  }
  
  // Validate axes
  if (perm_axes.size() != _shape.size()) {
      throw std::invalid_argument("Number of axes must match tensor dimensions");
  }
  
  // Check for duplicates and valid range
  std::vector<bool> used(_shape.size(), false);
  for (size_t axis : perm_axes) {
      if (axis >= _shape.size()) {
          throw std::out_of_range("Axis index out of range");
      }
      if (used[axis]) {
          throw std::invalid_argument("Duplicate axis in permutation");
      }
      used[axis] = true;
  }
  
  // Calculate new shape
  std::vector<size_t> new_shape(_shape.size());
  for (size_t i = 0; i < _shape.size(); ++i) {
      new_shape[i] = _shape[perm_axes[i]];
  }
  
  Tensor* result;
  if (inplace) {
      result = this;
  } else {
      result = copy(requires_grad);
  }
  
  // Create temporary storage for the transposed data
  std::vector<double> temp_storage(_storage.size());
  
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
  for (size_t i = 0; i < _storage.size(); ++i) {
      // Get original indices
      std::vector<size_t> orig_indices = flat_to_indices(i, _shape);
      
      // Create new indices based on permutation
      std::vector<size_t> new_indices(_shape.size());
      for (size_t j = 0; j < _shape.size(); ++j) {
          new_indices[j] = orig_indices[perm_axes[j]];
      }
      
      // Calculate new flat index and copy data
      size_t new_flat_idx = indices_to_flat(new_indices, new_shape);
      temp_storage[new_flat_idx] = _storage[i];
  }
  
  // Update shape and storage
  result->_shape = new_shape;
  result->_storage = std::move(temp_storage);
  
  return result;
}

Tensor* Tensor::squeeze(bool inplace, bool requires_grad) {
  std::vector<size_t> newshape; 
  for (auto s : this->shape()) {
    if (s != 1) {
      newshape.push_back(s);
    }
  }
  if (newshape.size() == 0) {
    newshape.push_back(1);
  } 
  return this->reshape(newshape, inplace, requires_grad);
}

Tensor* Tensor::squeeze(size_t dim, bool inplace, bool requires_grad) { 
  if (this->_shape[dim] != 1) {
    throw std::logic_error("The dimension you are squeezing is not 1.");
  }
  std::vector<size_t> newshape; 
  for (int i = 0; i < this->hdim(); i++) { 
    if (i != dim) {
      newshape.push_back(this->shape()[i]);
    }
  }
  return this->reshape(newshape, inplace, requires_grad); 
}

Tensor* Tensor::unsqueeze(size_t dim, bool inplace, bool requires_grad) { 
  // dim = dimension that you want to add the 1 in 
  std::vector<size_t> newshape; 
  for (int i = 0; i < this->hdim(); i++) {
    if (i == dim) {
      newshape.push_back(1); 
    }
    newshape.push_back(this->shape()[i]); 
  }
  return this->reshape(newshape, inplace, requires_grad); 
}

