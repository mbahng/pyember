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

std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape);
std::vector<size_t> concat_indices(std::vector<size_t> shape1, std::vector<size_t> shape2);
std::vector<size_t> duplicate_indices(std::vector<size_t> shape);

Tensor* Tensor::add(Tensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  size_t length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other->data()[i];
  }
  Tensor* out = new Tensor(res_data, this->shape()); 

  Tensor* this_ptr = this;
  Tensor* other_ptr = other; 

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr, out] {
    std::vector<size_t> newshape = concat_indices(out->shape(), this->shape());
    this->grad = new GradTensor(newshape, out->shape().size()); 
    other_ptr->grad = new GradTensor(newshape, out->shape().size()); 
    // update gradient to form d out[i]/ d this[i] = 1.  
    // Given sizes of (x_1, ..., x_N), we concat it to get (x_1..x_N, x_1..x_N) 
    for (std::vector<size_t> l_idx : generate_all_indices(this->shape())) {
      for (std::vector<size_t> r_idx : generate_all_indices(other_ptr->shape())) {
        std::vector<size_t> idx = concat_indices(l_idx, r_idx);
        if (l_idx == r_idx) { 
          (this->grad)->at(idx) = 1.0;
          (other_ptr->grad)->at(idx) = 1.0;
        }
        else {
          (this->grad)->at(idx) = 0.0;
          (other_ptr->grad)->at(idx) = 0.0;
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::add(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other->data()[i];
  }

  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::add(ScalarTensor* other) {
  return other->add(this); 
}

Tensor* Tensor::add(double* other) {
  // Must store scalar on heap since we don't want it destroyed in comp graph
  ScalarTensor* scalar = new ScalarTensor(*other);
  return scalar->add(this); 
}

Tensor* Tensor::sub(Tensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  size_t length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other->data()[i];
  }
  Tensor* out = new Tensor(res_data, this->shape()); 

  Tensor* this_ptr = this;
  Tensor* other_ptr = other;

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr, out] {
    std::vector<size_t> newshape = concat_indices(out->shape(), this->shape());
    this->grad = new GradTensor(newshape, this->shape().size()); 
    other_ptr->grad = new GradTensor(newshape, this->shape().size()); 
    // update gradient to form d out[i]/ d this[i] = 1.  
    // Given sizes of (x_1, ..., x_N), we concat it to get (x_1..x_N, x_1..x_N) 
    for (std::vector<size_t> l_idx : generate_all_indices(other_ptr->shape())) {
      for (std::vector<size_t> r_idx : generate_all_indices(this->shape())) {
        std::vector<size_t> idx = concat_indices(l_idx, r_idx);
        if (l_idx == r_idx) {
          (this->grad)->at(idx) = 1.0;
          (other_ptr->grad)->at(idx) = -1.0;
        }
        else {
          (this->grad)->at(idx) = 0.0;
          (other_ptr->grad)->at(idx) = 0.0;
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::sub(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other->data()[i];
  }
  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::sub(ScalarTensor* other) {
  return other->sub(this); 
}

Tensor* Tensor::sub(double* other) { 
  ScalarTensor* scalar = new ScalarTensor(*other);
  return scalar->sub(this); 
}

Tensor* Tensor::mul(Tensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  size_t length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other->data()[i];
  }
  Tensor* out = new Tensor(res_data, this->shape()); 

  Tensor* this_ptr = this;
  Tensor* other_ptr = other;
  
  out->prev = {this_ptr, other_ptr};
  out->grad = new GradTensor(duplicate_indices(this->shape()), (this->shape()).size());

  out->backward = [this, other_ptr, out] {
    std::vector<size_t> newshape = concat_indices(out->shape(), this->shape());
    this->grad = new GradTensor(newshape, this->shape().size()); 
    other_ptr->grad = new GradTensor(newshape, this->shape().size()); 
    // update gradient to form d out[i]/ d this[i] = 1.  
    // Given sizes of (x_1, ..., x_N), we concat it to get (x_1..x_N, x_1..x_N) 
    std::vector<size_t> pairshape = duplicate_indices(this->shape_);

    for (std::vector<size_t> l_idx : generate_all_indices(this->shape_)) {
      for (std::vector<size_t> r_idx : generate_all_indices(other_ptr->shape_)) { 
        if (l_idx == r_idx) {
          std::vector<size_t> idx = concat_indices(l_idx, r_idx);
          (this->grad)->at(idx) = other_ptr->at(r_idx);
          (other_ptr->grad)->at(idx) = this->at(l_idx);
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::mul(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other->data()[i];
  }
  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::mul(ScalarTensor* other) {
  return other->mul(this); 
}

Tensor* Tensor::mul(double* other) {
  ScalarTensor* scalar = new ScalarTensor(*other);
  return scalar->mul(this); 
}

Tensor* Tensor::matmul(Tensor* other) {
  // Check if the tensors are at least 2D
  assert(this->shape().size() == 2 || other->shape().size() == 2);
  // Check if the last dimension of this tensor matches the second-to-last dimension of other
  assert(this->shape()[1] == other->shape()[0]);
  // Determine the dimensions of the result
  std::vector<size_t> result_shape {this->shape()[0], other->shape()[1]};
  Tensor* out = new Tensor(std::vector<double> (shape_to_length(result_shape), 0.0), result_shape);
  
  // Perform batch matrix multiplication
  size_t m = this->shape()[0];
  size_t n = this->shape()[1];
  size_t p = other->shape()[1];
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += this->data()[i * n + k] * other->data()[k * p + j];
      }
      out->storage_[i * p + j] = sum;
    }
  }

  Tensor* this_ptr = this;
  Tensor* other_ptr = other;

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr, m, n, p] {
    // Set up gradients with correct shapes
    // For A: shape is (M, K, M, N) where M,K is output shape and M,N is input shape
    std::vector<size_t> left_grad_shape = {m, p, m, n};
    // For B: shape is (M, K, N, K) where M,K is output shape and N,K is input shape
    std::vector<size_t> right_grad_shape = {m, p, n, p};
    
    this->grad = new GradTensor(left_grad_shape, 2);  // pivot after output dimensions
    other_ptr->grad = new GradTensor(right_grad_shape, 2); // pivot after output dimensions
    // For the left matrix (A):
    // ∂C[i,j]/∂A[k,l] = B[l,j] if i=k, 0 otherwise
    for (size_t i = 0; i < m; ++i) {     // output row
      for (size_t j = 0; j < p; ++j) {   // output col
        for (size_t k = 0; k < m; ++k) { // input A row
          for (size_t l = 0; l < n; ++l) { // input A col
            std::vector<size_t> grad_idx = {i, j, k, l};
            if (i == k) {
              (this->grad)->at(grad_idx) = other_ptr->data()[l * p + j];
            } else {
              (this->grad)->at(grad_idx) = 0.0;
            }
          }
        }
      }
    }

    // For the right matrix (B):
    // ∂C[i,j]/∂B[k,l] = A[i,k] if j=l, 0 otherwise
    for (size_t i = 0; i < m; ++i) {     // output row
      for (size_t j = 0; j < p; ++j) {   // output col
        for (size_t k = 0; k < n; ++k) { // input B row
          for (size_t l = 0; l < p; ++l) { // input B col
            std::vector<size_t> grad_idx = {i, j, k, l};
            if (j == l) {
              (other_ptr->grad)->at(grad_idx) = this->data()[i * n + k];
            } else {
              (other_ptr->grad)->at(grad_idx) = 0.0;
            }
          }
        }
      }
    }
  };

  return out;
}


