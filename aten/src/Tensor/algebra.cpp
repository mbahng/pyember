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
    std::vector<size_t> newshape = concat( 
      this_ptr->b_indices(), 
      other->b_indices(), 
      this_ptr->nb_indices(), 
      other->nb_indices()
    ); 
    size_t bidx = this_ptr->bidx() + other->bidx(); 
    size_t pidx = this_ptr->shape().size() + other->b_indices().size(); 

    // fill in gradients for this 
    if (this_ptr->has_grad) {

      this_ptr->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
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
      other->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
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
        out->at(concat(b1, b2, i)) = this->at(concat(b1, i)) - other->at(concat(b2, i));
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
    std::vector<size_t> newshape = concat( 
      this_ptr->b_indices(), 
      other->b_indices(), 
      this_ptr->nb_indices(), 
      other->nb_indices()
    ); 
    size_t bidx = this_ptr->bidx() + other->bidx(); 
    size_t pidx = this_ptr->shape().size() + other->b_indices().size(); 

    // fill in gradients for this 
    if (this_ptr->has_grad) {

      this_ptr->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
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
      other->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
              if (l_idx == r_idx) { 
                (other->grad)->at(idx) = -1.0;
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
        out->at(concat(b1, b2, i)) = this->at(concat(b1, i)) * other->at(concat(b2, i));
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
    std::vector<size_t> newshape = concat( 
      this_ptr->b_indices(), 
      other->b_indices(), 
      this_ptr->nb_indices(), 
      other->nb_indices()
    ); 
    size_t bidx = this_ptr->bidx() + other->bidx(); 
    size_t pidx = this_ptr->shape().size() + other->b_indices().size(); 

    // fill in gradients for this 
    if (this_ptr->has_grad) {

      this_ptr->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
              if (l_idx == r_idx) { 
                (this_ptr->grad)->at(idx) = other->at(concat(b2, r_idx));
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
      other->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->nb_indices())) {
            for (std::vector<size_t> r_idx : generate_all_indices(other->nb_indices())) {
              std::vector<size_t> idx = concat(b1, b2, l_idx, r_idx);
              if (l_idx == r_idx) { 
                (other->grad)->at(idx) = this_ptr->at(concat(b1, l_idx));
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
  if (this->nb_indices().size() != 2 || other->nb_indices().size() != 2) { 
    std::string this_shape = ""; 
    for (auto s : this->shape()) { this_shape += " " + std::to_string(s); } 
    std::string other_shape = ""; 
    for (auto s : other->shape()) { other_shape += " " + std::to_string(s); } 
    throw std::logic_error("Only rank-2 tensors x rank-2 tensors are supported. \n"
        "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
  }
  if (this->nb_indices()[1] != other->nb_indices()[0]) {
    std::string this_shape = ""; 
    for (auto s : this->nb_indices()) { this_shape += " " + std::to_string(s); } 
    std::string other_shape = ""; 
    for (auto s : other->nb_indices()) { other_shape += " " + std::to_string(s); } 
    throw std::logic_error("The dimension of the contracted rank does not match. \n"
        "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
  }

  std::vector<size_t> contracted_shape = {this->nb_indices()[0], other->nb_indices()[1]};

  // Determine the dimensions of the result
  std::vector<size_t> result_shape = concat(
    this->b_indices(), 
    other->b_indices(),
    contracted_shape
  );
  size_t res_bidx = this->bidx() + other->bidx();
  Tensor* out = Tensor::zeros(result_shape, res_bidx);
  // set has_grad of output 
  if (this->has_grad || other->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  std::vector<size_t> m = {this->nb_indices()[0]};
  std::vector<size_t> n = {this->nb_indices()[1]}; 
  std::vector<size_t> p = {other->nb_indices()[1]};

  for (std::vector<size_t> b1 : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) { 
      for (std::vector<size_t> i : generate_all_indices(m)) {
        for (std::vector<size_t> j : generate_all_indices(p)) {
          double sum = 0.0; 
          for (std::vector<size_t> k : generate_all_indices(n)) {
            sum += this->at(concat(b1, i, k)) * other->at(concat(b2, k, j)); 
          }
          out->at(concat(b1, b2, i, j)) = sum;
        }
      }
    }
  }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
  if (other->has_grad) { out->prev.push_back(other); } 

  Tensor* this_ptr = this; 

  out->backward = [this_ptr, other] { 
    std::vector<size_t> m = {this_ptr->nb_indices()[0]};
    std::vector<size_t> n = {this_ptr->nb_indices()[1]}; 
    std::vector<size_t> p = {other->nb_indices()[1]};

    std::vector<size_t> left_grad_shape = concat(
      this_ptr->b_indices(), 
      other->b_indices(), 
      m, p, m, n
    );
    std::vector<size_t> right_grad_shape = concat(
      this_ptr->b_indices(), 
      other->b_indices(), 
      m, p, n, p
    ); 

    size_t bidx = this_ptr->b_indices().size() + other->b_indices().size();
    size_t pidx = this_ptr->shape().size() + other->b_indices().size(); 

    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(left_grad_shape, pidx, bidx);  
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> i : generate_all_indices(m)) {
            for (std::vector<size_t> j : generate_all_indices(p)) {
              for (std::vector<size_t> k : generate_all_indices(n)) {
                (this_ptr->grad)->at(concat(b1, b2, i, j, i, k)) = other->at(concat(b2, k, j));
              }
            }
          }
        }
      }

    }
    if (other->has_grad) {
      other->grad = new GradTensor(right_grad_shape, pidx, bidx);  
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> i : generate_all_indices(m)) {
            for (std::vector<size_t> j : generate_all_indices(p)) {
              for (std::vector<size_t> k : generate_all_indices(n)) {
                (other->grad)->at(concat(b1, b2, i, j, k, j)) = this_ptr->at(concat(b1, i, k));
              }
            }
          }
        }
      }
    }
  };

  return out;
}


