#include "../Tensor.h"
#include <vector>
#include <cassert>
#include <cmath>

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

Tensor* Tensor::sum() { 
  double out_data = 0.0;
  size_t length = shape_to_length(this->shape());
  for (int i = 0; i < length; ++i) {
    out_data += this->data()[i]; 
  }

  Tensor* out = new Tensor({out_data}, std::vector<size_t>{1}); 
  // set has_grad of output 
  if (this->has_grad) { out->has_grad = true; }
  else { out->has_grad = false; }

  out->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { out->prev.push_back(this); }
 
  out->backward = [this] { 
    if (this->has_grad) {
      std::vector<size_t> newshape = concat_indices({1}, this->shape());
      this->grad = new GradTensor(newshape, 1);  
      for (size_t i = 0; i < ((this->grad)->storage_).size(); ++i) {
        (this->grad)->storage_[i] = 1.0;
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

  out->backward = [this, x_ptr] {  
    if (this->has_grad) {
      std::vector<size_t> newshape = duplicate_indices(this->shape());
      this->grad = new GradTensor(newshape, (this->shape()).size());  
      for (std::vector<size_t> l_idx : generate_all_indices(this->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this->shape())) {
          std::vector<size_t> idx = concat_indices(l_idx, r_idx); 
          if (l_idx == r_idx) {
            (this->grad)->at(idx) = *x_ptr * std::pow(this->at(l_idx), (*x_ptr)-1);  
          }
          else {
            (this->grad)->at(idx) = 0.0; 
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

  out->backward = [this] { 
    if (this->has_grad) {
      std::vector<size_t> newshape = duplicate_indices(this->shape());
      this->grad = new GradTensor(newshape, (this->shape()).size());  
      for (std::vector<size_t> l_idx : generate_all_indices(this->shape())) {
        for (std::vector<size_t> r_idx : generate_all_indices(this->shape())) {
          std::vector<size_t> idx = concat_indices(l_idx, r_idx); 
          if ((l_idx == r_idx) && (this->at(l_idx) >= 0.0)) {
            (this->grad)->at(idx) = 1.0; 
          }
          else {
            (this->grad)->at(idx) = 0.0; 
          }
        }
      }
    }
  };
  return out; 
}


