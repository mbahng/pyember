#include "../Tensor.h"
#include "../../Util/utils.h"

GradTensor* GradTensor::shallowcopy() const {
  // creates a shallow copy 
  return new GradTensor(this->_storage, this->_shape, this->bidx, this->pidx()); 
}

GradTensor* GradTensor::deepcopy() const {
  // creates a deep copy 
  std::vector<double> storage = this->_storage; 
  std::vector<size_t> shape = this->_shape;  
  size_t bidx = this->bidx; 
  size_t pidx = this->pidx();
  return new GradTensor(storage, shape, bidx, pidx); 
}

GradTensor* GradTensor::copy() const {
  return this->shallowcopy();
}

GradTensor* GradTensor::reshape(std::vector<size_t> new_shape, bool inplace) {

  if (CIntegrity::prod(new_shape) != this->size()) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  if (inplace) { 
    // should not be used in the computational graph
    this->_shape = new_shape; 
    return this; 
  }
  else { 
    // usually pidx will become meaningless when you reshape it. 
    GradTensor* out = new GradTensor(_storage, new_shape, bidx, _pidx);
    return out; 
  }
} 

GradTensor* GradTensor::transpose(size_t d1, size_t d2) {
  if (d1 >= this->rank() or d2 >= this->rank()) {
    throw std::invalid_argument("Transposed ranks are out of bounds.");
  }

  size_t res_bidx = this->bidx;
  size_t res_pidx = this->pidx();

  if (d1 < this->bidx or d2 < this->bidx) {
    res_bidx = 0; 
    std::cerr << "You are attempting to transpose the batch ranks. The result bidx will be reset to 0.\n";
  }

  if ((d1 < this->pidx() && d2 >= this->pidx()) || (d1 >= this->pidx() && d2 < this->pidx())) {
    std::cerr << "You are attempting to transpose ranks across the pivot. The pivot is set to the original gradtensor pivot. Proceed with caution.\n"; 
  }

  // newshape is divided into 5 parts: before d1, d1, between, d2, after d2 
  std::vector<size_t> before = std::vector<size_t>(this->shape().begin(), this->shape().begin() + d1);  
  std::vector<size_t> d1_idx = std::vector<size_t>{this->shape()[d1]};
  std::vector<size_t> between = std::vector<size_t>(this->shape().begin() + d1 + 1, this->shape().begin() + d2); 
  std::vector<size_t> d2_idx = std::vector<size_t>{this->shape()[d2]};
  std::vector<size_t> after = std::vector<size_t>(this->shape().begin() + d2 + 1, this->shape().end());

  GradTensor* res = new GradTensor(
    Index::concat(before, d2_idx, between, d1_idx, after), 
    res_bidx, res_pidx
  );

  for (auto bef : Index::generate_all_indices(before)) {
    for (auto _d1 : Index::generate_all_indices(d1_idx)) {
      for (auto bet : Index::generate_all_indices(between)) {
        for (auto _d2 : Index::generate_all_indices(d2_idx)) {
          for (auto aft : Index::generate_all_indices(after)) {
            res->at(Index::concat(bef, _d2, bet, _d1, aft)) = this->at(Index::concat(bef, _d1, bet, _d2, aft));
          }
        }
      }
    }
  }
  return res; 
}

GradTensor* GradTensor::transpose() { 
  return this->transpose(this->rank() - 2, this->rank() - 1);  
}

GradTensor* GradTensor::transpose(const std::vector<size_t>& axes) {
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
      
      // Calculate new flat index and copy data to temporary storage
      size_t new_flat_idx = indices_to_flat(new_indices, new_shape);
      temp_storage[new_flat_idx] = _storage[i];
  }
  
  // Update shape and storage
  _shape = new_shape;
  _storage = std::move(temp_storage);
  
  return this;
}


