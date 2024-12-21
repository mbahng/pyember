#include <vector>
#include <cassert>
#include "../Tensor.h"
#include "../utils.h"

Tensor* Tensor::add(Tensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match");
  } 

  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());
  size_t new_bidx = this->bidx() + other->bidx(); 
  std::vector<double> res_data(prod(new_shape), 0.0); 
  Tensor* out = new Tensor(res_data, new_shape, new_bidx); 

  // fill in data in batches
  for (std::vector<size_t> b1 : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
      for (std::vector<size_t> i : generate_all_indices(this->nb_indices())) {
        out->at(concat(b1, b2, i)) = this->at(concat(b1, i)) + other->at(concat(b2, i));
      }
    }
  }

  // set has_grad of output 
  if (this->has_grad || other->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
  if (other->has_grad) { out->prev.push_back(other); }

  Tensor* this_ptr = this; 

  out->backward = [this_ptr, other, out] { 

    // fill in gradients for this 
    if (this_ptr->has_grad) {
      std::vector<size_t> this_newshape = concat( 
        this_ptr->b_indices(), 
        other->b_indices(), 
        this_ptr->b_indices(), 
        this_ptr->nb_indices(), 
        other->nb_indices()
      ); 
      size_t this_bidx = this_ptr->bidx() + other->bidx() + this_ptr->bidx(); 
      size_t this_pidx = 2 * this_ptr->b_indices().size() + other->bidx() + this_ptr->nb_indices().size();

      this_ptr->grad = new GradTensor(this_newshape, this_pidx, this_bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, b1, l_idx, r_idx);
              if (l_idx == r_idx) { 
                (this_ptr->grad)->at(idx) = 1.0;
              }
              else {
                (this_ptr->grad)->at(idx) = 0.0;
              }
            }
          }
        }
      }
    }
    // fill in gradients for other 
    if (other->has_grad) {
      std::vector<size_t> other_newshape = concat( 
        this_ptr->b_indices(), 
        other->b_indices(), 
        other->b_indices(), 
        this_ptr->nb_indices(), 
        other->nb_indices()
      ); 
      size_t other_bidx = this_ptr->bidx() + other->bidx() + other->bidx(); 
      size_t other_pidx = 2 * other->b_indices().size() + this_ptr->bidx() + other->nb_indices().size();

      other->grad = new GradTensor(other_newshape, other_pidx, other_bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, b2, l_idx, r_idx);
              if (l_idx == r_idx) { 
                (other->grad)->at(idx) = 1.0;
              }
              else {
                (other->grad)->at(idx) = 0.0;
              }
            }
          }
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
  // set has_grad of output 
  if (this->has_grad || other->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
  if (other->has_grad) { out->prev.push_back(other); } 

  Tensor* this_ptr = this; 

  out->backward = [this_ptr, other, out] {
    std::vector<size_t> newshape = concat(out->shape(), this_ptr->shape());
    
    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, this_ptr->shape().size()); 
      for (std::vector<size_t> l_idx : generate_all_indices(other->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this_ptr->shape())) {
          std::vector<size_t> idx = concat(l_idx, r_idx);
          if (l_idx == r_idx) {
            (this_ptr->grad)->at(idx) = 1.0;
          }
          else {
            (this_ptr->grad)->at(idx) = 0.0;
          }
        }
      }
    }
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, this_ptr->shape().size()); 
      for (std::vector<size_t> l_idx : generate_all_indices(other->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this_ptr->shape())) {
          std::vector<size_t> idx = concat(l_idx, r_idx);
          if (l_idx == r_idx) {
            (other->grad)->at(idx) = -1.0;
          }
          else {
            (other->grad)->at(idx) = 0.0;
          }
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
  // set has_grad of output 
  if (this->has_grad || other->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
  if (other->has_grad) { out->prev.push_back(other); } 

  Tensor* this_ptr = this; 

  out->backward = [this_ptr, other, out] {
    std::vector<size_t> newshape = concat(out->shape(), this_ptr->shape()); 

    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, this_ptr->shape().size()); 
      std::vector<size_t> pairshape = duplicate(this_ptr->shape_);

      for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->shape_)) {
        for (std::vector<size_t> r_idx : generate_all_indices(other->shape_)) { 
          std::vector<size_t> idx = concat(l_idx, r_idx);
          if (l_idx == r_idx) {
            (this_ptr->grad)->at(idx) = other->at(r_idx);
          }
          else {
            (this_ptr->grad)->at(idx) = 0.0;
          }
        }
      }
    }
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, other->shape().size()); 
      std::vector<size_t> pairshape = duplicate(this_ptr->shape_);

      for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->shape_)) {
        for (std::vector<size_t> r_idx : generate_all_indices(other->shape_)) { 
          std::vector<size_t> idx = concat(l_idx, r_idx);
          if (l_idx == r_idx) {
            (other->grad)->at(idx) = this_ptr->at(l_idx);
          }
          else {
            (other->grad)->at(idx) = 0.0;
          }
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
  if (this->shape().size() != 2 || other->shape().size() != 2) { 
    std::string this_shape = ""; 
    for (auto s : this->shape()) { this_shape += " " + std::to_string(s); } 
    std::string other_shape = ""; 
    for (auto s : other->shape()) { other_shape += " " + std::to_string(s); } 
    throw std::logic_error("Only rank-2 tensors x rank-2 tensors are supported. \n"
        "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
  }
  if (this->shape()[1] != other->shape()[0]) {
    std::string this_shape = ""; 
    for (auto s : this->shape()) { this_shape += " " + std::to_string(s); } 
    std::string other_shape = ""; 
    for (auto s : other->shape()) { other_shape += " " + std::to_string(s); } 
    throw std::logic_error("The dimension of the contracted rank does not match. \n"
        "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
  }
  // Determine the dimensions of the result
  std::vector<size_t> result_shape {this->shape()[0], other->shape()[1]};
  Tensor* out = new Tensor(std::vector<double> (shape_to_length(result_shape), 0.0), result_shape);
  // set has_grad of output 
  if (this->has_grad || other->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

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

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
  if (other->has_grad) { out->prev.push_back(other); } 

  Tensor* this_ptr = this; 

  out->backward = [this_ptr, other, m, n, p] {
    std::vector<size_t> left_grad_shape = {m, p, m, n};
    std::vector<size_t> right_grad_shape = {m, p, n, p}; 

    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(left_grad_shape, 2);  
      for (size_t i = 0; i < m; ++i) {     
        for (size_t j = 0; j < p; ++j) {   
          for (size_t k = 0; k < m; ++k) { 
            for (size_t l = 0; l < n; ++l) { 
              std::vector<size_t> grad_idx = {i, j, k, l};
              if (i == k) {
                (this_ptr->grad)->at(grad_idx) = other->data()[l * p + j];
              } else {
                (this_ptr->grad)->at(grad_idx) = 0.0;
              }
            }
          }
        }
      }
    }
    if (other->has_grad) {
      other->grad = new GradTensor(right_grad_shape, 2); 
      for (size_t i = 0; i < m; ++i) {     
        for (size_t j = 0; j < p; ++j) {  
          for (size_t k = 0; k < n; ++k) { 
            for (size_t l = 0; l < p; ++l) { 
              std::vector<size_t> grad_idx = {i, j, k, l};
              if (j == l) {
                (other->grad)->at(grad_idx) = this_ptr->data()[i * n + k];
              } else {
                (other->grad)->at(grad_idx) = 0.0;
              }
            }
          }
        }
      }
    }
  };

  return out;
}


