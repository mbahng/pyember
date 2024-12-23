#include "../Tensor.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm> 
#include "../utils.h"

Tensor* Tensor::dot(Tensor* other) { 
  Integrity::Shape r = Integrity::compat(this, other); 

  std::vector<size_t> new_shape = concat(r.b_shape, std::vector<size_t>{1});
  std::vector<double> res_data(prod(new_shape), 0.0); 
  Tensor* res = new Tensor(new_shape, std::max(this->bidx_, other->bidx_)); 

  if (this->shape().size() >= other->shape().size()) {
    for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
      double dot_product = 0.0; 
      for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
        dot_product += this->at(concat(b, i)) * other->at(i); 
      }
      res->at(concat(b, std::vector<size_t>{0})) = dot_product; 
    }
  }
  else {
    for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
      double dot_product = 0.0; 
      for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
        dot_product += this->at(i) * other->at(concat(b, i)); 
      }
      res->at(concat(b, std::vector<size_t>{0})) = dot_product; 
    }
  }


  // set has_grad of output 
  if (this->has_grad || other->has_grad) { res->has_grad = true; }
  else { res->has_grad = false; }

  res->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { res->prev.push_back(this); }
  if (other->has_grad) { res->prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->backward = [this_ptr, other, res, r] { 
    std::vector<size_t> newshape = concat(r.b_shape, std::vector<size_t>{1}, r.nb_shape); 
    size_t bidx = std::max(this_ptr->bidx_, other->bidx_); 
    size_t pidx = bidx + 1 ; 

    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, bidx, pidx);  
      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(concat(b, std::vector<size_t>{0}, i)) = other->at(i);
          }
        }
      }
      else {
        for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(concat(b, std::vector<size_t>{0}, i)) = other->at(concat(b, i));
          }
        }
      }
    }

    if (other->has_grad) {
      other->grad = new GradTensor(newshape, bidx, pidx); 
      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
            (other->grad)->at(concat(b, std::vector<size_t>{0}, i)) = this_ptr->at(concat(b, i));
          }
        }
      }
      else{
        for (std::vector<size_t> b : generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : generate_all_indices(r.nb_shape)) {
            (other->grad)->at(concat(b, std::vector<size_t>{0}, i)) = this_ptr->at(i);
          }
        }
      }
    }
  };
  return res; 
}

Tensor* Tensor::sum() { 
  double out_data = 0.0;
  size_t length = prod(this->shape());
  for (int i = 0; i < length; ++i) {
    out_data += this->data()[i]; 
  }

  Tensor* out = new Tensor({out_data}, std::vector<size_t>{1}); 
  // set has_grad of output 
  if (this->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); } 

  Tensor* this_ptr = this; 
 
  out->backward = [this_ptr] { 
    if (this_ptr->has_grad) {
      std::vector<size_t> newshape = concat(this_ptr->bshape(), {1}, this_ptr->nbshape());
      this_ptr->grad = new GradTensor(newshape, this_ptr->bidx_, this_ptr->bidx_ + 1);  
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
      std::vector<size_t> newshape = concat(this_ptr->shape(), this_ptr->nbshape());
      this_ptr->grad = new GradTensor(newshape, this_ptr->bidx_, (this_ptr->shape()).size());  

      for (std::vector<size_t> b : generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(concat(b, i, i)) = *x_ptr * std::pow(this_ptr->at(concat(b, i)), (*x_ptr)-1);  
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
      this_ptr->grad = new GradTensor(newshape, this_ptr->bidx_, (this_ptr->shape()).size());  
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


