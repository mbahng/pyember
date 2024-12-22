#include "../Tensor.h"
#include <vector>
#include <cassert>
#include <cmath>
#include "../utils.h"

Tensor* Tensor::dot(Tensor* other) { 
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match");
  } 

  std::vector<size_t> new_shape = concat(this->b_indices(), other->b_indices(), std::vector<size_t>{1});
  size_t new_bidx = this->bidx() + other->bidx(); 
  std::vector<double> res_data(prod(new_shape), 0.0); 
  Tensor* out = new Tensor(res_data, new_shape, new_bidx); 

  // fill in data in batches
  for (std::vector<size_t> b1 : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) { 
      double sum = 0.0; 
      for (std::vector<size_t> i : generate_all_indices(this->nb_indices())) {
        sum += this->at(concat(b1, i)) * other->at(concat(b2, i)); 
      }
      out->at(concat(b1, b2, std::vector<size_t>{0})) = sum; 
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
      std::vector<size_t>{1}, 
      other->nb_indices()
    ); 
    size_t bidx = this_ptr->bidx() + other->bidx(); 
    size_t pidx = bidx + 1; 

    // fill in gradients for this 
    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> idx : generate_all_indices(this_ptr->nb_indices())) {
            (this_ptr->grad)->at(concat(b1, b2, std::vector<size_t>{0}, idx)) = other->at(concat(b2, idx));
          }
        }
      }
    }
    // fill in gradients for other 
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, pidx, bidx); 
      for (std::vector<size_t> b1 : generate_all_indices(this_ptr->b_indices())) {
        for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
          for (std::vector<size_t> idx : generate_all_indices(this_ptr->nb_indices())) {
            (other->grad)->at(concat(b1, b2, std::vector<size_t>{0}, idx)) = this_ptr->at(concat(b1, idx));
          }
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::sum() { 
  double out_data = 0.0;
  size_t length = shape_to_length(this->shape());
  for (int i = 0; i < length; ++i) {
    out_data += this->data()[i]; 
  }

  Tensor* out = new Tensor({out_data}, std::vector<size_t>{1, 1}); 
  // set has_grad of output 
  if (this->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); } 

  Tensor* this_ptr = this; 
 
  out->backward = [this_ptr] { 
    if (this_ptr->has_grad) {
      std::vector<size_t> newshape = concat({1, 1}, this_ptr->shape());
      this_ptr->grad = new GradTensor(newshape, 2);  
      for (size_t i = 0; i < ((this_ptr->grad)->storage_).size(); ++i) {
        (this_ptr->grad)->storage_[i] = 1.0;
      }
    }
  };
  return out; 
}

Tensor* Tensor::pow(double* x) { 
  Tensor* out = this->copy(); 
  if (this->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  for (int i = 0; i < (this->storage_).size(); i++) {
    (out->storage_)[i] = std::pow((this->storage_)[i], *x);
  }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }

  // need to allocate this x on heap for it to be accessible by backward
  double* x_ptr = new double(2); 
  
  Tensor* this_ptr = this; 
  out->backward = [this_ptr, x_ptr] {  
    if (this_ptr->has_grad) {
      std::vector<size_t> newshape = duplicate(this_ptr->shape());
      this_ptr->grad = new GradTensor(newshape, (this_ptr->shape()).size());  
      for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this_ptr->shape())) {
          std::vector<size_t> idx = concat(l_idx, r_idx); 
          if (l_idx == r_idx) {
            (this_ptr->grad)->at(idx) = *x_ptr * std::pow(this_ptr->at(l_idx), (*x_ptr)-1);  
          }
          else {
            (this_ptr->grad)->at(idx) = 0.0; 
          }
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::relu() { 
  Tensor* out = this->copy(); 
  if (this->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  for (int i = 0; i < (this->storage_).size(); i++) {
    if ((this->storage_)[i] < 0.0) { 
      (out->storage_)[i] = 0.0;
    }
  }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }

  Tensor* this_ptr = this;  

  out->backward = [this_ptr] { 
    if (this_ptr->has_grad) {
      std::vector<size_t> newshape = duplicate(this_ptr->shape());
      this_ptr->grad = new GradTensor(newshape, (this_ptr->shape()).size());  
      for (std::vector<size_t> l_idx : generate_all_indices(this_ptr->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this_ptr->shape())) {
          std::vector<size_t> idx = concat(l_idx, r_idx); 
          if ((l_idx == r_idx) && (this_ptr->at(l_idx) >= 0.0)) {
            (this_ptr->grad)->at(idx) = 1.0; 
          }
          else {
            (this_ptr->grad)->at(idx) = 0.0; 
          }
        }
      }
    }
  };
  return out; 
}


