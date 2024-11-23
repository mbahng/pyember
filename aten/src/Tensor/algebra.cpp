#include <vector>
#include <cassert>
#include "../Tensor.h"

int shape_to_length(std::vector<size_t> shape); 

void array_matches_shape(
  std::vector<double> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
);
void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
);

// Helper function to increment the indices
bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    indices[i]++;
    if (indices[i] < shape[i]) {
      return true;  // Successfully incremented
    }
    indices[i] = 0;  // Reset this position and continue with next position
  }
  return false;  // We've gone through all possibilities
}

// Function to generate all possible vectors
std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape) {
  std::vector<std::vector<size_t>> result;
  
  // Calculate total number of combinations
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }
  result.reserve(total);  // Reserve space for efficiency
  
  // Start with all zeros
  std::vector<size_t> current(shape.size(), 0);
  
  // Add first combination
  result.push_back(current);
  
  // Generate all other combinations
  while (increment_indices(current, shape)) {
    result.push_back(current);
  }
  
  return result;
}

Tensor Tensor::add(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  size_t length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }
  Tensor out = Tensor(res_data, this->shape()); 

  std::vector<size_t> newshape = out.shape(); 
  newshape.insert(newshape.end(), this->shape().begin(), this->shape().end()); 
  this->grad = GradTensor(newshape, this->shape().size()); 
  other.grad = GradTensor(newshape, this->shape().size()); 

  Tensor* this_ptr = this;
  Tensor* other_ptr = &other;

  out.prev = {this_ptr, other_ptr};

  out.backward = [this, other_ptr] {
    // update gradient to form d out[i]/ d this[i] = 1.  
    // Given sizes of (x_1, ..., x_N), we concat it to get (x_1..x_N, x_1..x_N) 
    std::vector<size_t> pairshape = this->shape_; 
    pairshape.insert(pairshape.end(), (this->shape_).begin(), (this->shape_).end());
    // Generate a meshgrid over all indices, can be very inefficient
    auto relevant_idx = generate_all_indices(pairshape);
    for (std::vector<size_t> idx : relevant_idx) { 
      // For each index, we must either set it as 1 if 1st half = 2nd half
      bool same_idx = true;
      for (int i = 0; i < idx.size() / 2; i++) {
        if (idx[i] != idx[i + (idx.size() / 2)]) {
          same_idx = false;
        }
      }
      if (same_idx) {
        (this->grad).at(idx) = 1.0;
        (other_ptr->grad).at(idx) = 1.0;
      }
      else {
        (this->grad).at(idx) = 0.0;
        (other_ptr->grad).at(idx) = 0.0;
      }
    }
  };
  return out; 
}

Tensor Tensor::add(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return Tensor(res_data, this->shape());   
}

Tensor Tensor::sub(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  size_t length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }
  Tensor out = Tensor(res_data, this->shape()); 

  std::vector<size_t> newshape = out.shape(); 
  newshape.insert(newshape.end(), this->shape().begin(), this->shape().end()); 
  this->grad = GradTensor(newshape, this->shape().size()); 
  other.grad = GradTensor(newshape, this->shape().size()); 

  Tensor* this_ptr = this;
  Tensor* other_ptr = &other;

  out.prev = {this_ptr, other_ptr};

  out.backward = [this, other_ptr] {
    // update gradient to form d out[i]/ d this[i] = 1.  
    // Given sizes of (x_1, ..., x_N), we concat it to get (x_1..x_N, x_1..x_N) 
    std::vector<size_t> pairshape = this->shape_; 
    pairshape.insert(pairshape.end(), (this->shape_).begin(), (this->shape_).end());
    // Generate a meshgrid over all indices, can be very inefficient
    auto relevant_idx = generate_all_indices(pairshape);
    for (std::vector<size_t> idx : relevant_idx) { 
      // For each index, we must either set it as 1 if 1st half = 2nd half
      bool same_idx = true;
      for (int i = 0; i < idx.size() / 2; i++) {
        if (idx[i] != idx[i + (idx.size() / 2)]) {
          same_idx = false;
        }
      }
      if (same_idx) {
        (this->grad).at(idx) = 1.0;
        (other_ptr->grad).at(idx) = -1.0;
      }
      else {
        (this->grad).at(idx) = 0.0;
        (other_ptr->grad).at(idx) = 0.0;
      }
    }
  };
  return out; 
}

Tensor Tensor::sub(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::mul(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::mul(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }
  return Tensor(res_data, this->shape());   
}

Tensor Tensor::matmul(Tensor& other) {
  // Check if the tensors are at least 2D
  assert(this->shape().size() == 2 || other.shape().size() == 2);

  // Check if the last dimension of this tensor matches the second-to-last dimension of other
  assert(this->shape()[1] == other.shape()[0]);

  // Determine the dimensions of the result
  std::vector<size_t> result_shape {this->shape()[0], other.shape()[1]};

  Tensor out(std::vector<double> (shape_to_length(result_shape), 0.0), result_shape);

  // Perform batch matrix multiplication
  int m = this->shape()[0];
  int n = this->shape()[1];
  int p = other.shape()[1];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += this->data()[i * n + k] * other.data()[k * p + j];
      }
      out.storage_[i * p + j] = sum;
    }
  }
  return out;
}


