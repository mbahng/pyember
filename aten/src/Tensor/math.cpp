#include "../Tensor.h"
#include <vector>
#include <cassert>
#include <cmath>
#include "../utils.h"

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


