#include "../Tensor.h"
#include <vector>
#include "../utils.h"

Tensor* ScalarTensor::add(Tensor* other) {
  Tensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] += this->item();
  }

  std::vector<size_t> this_grad_shape
    = concat(out->shape(), this->shape()); 
  std::vector<size_t> other_grad_shape
    = concat(out->shape(), other->shape()); 


  this->grad = new GradTensor(this_grad_shape, other->bidx_, out->shape().size());
  other->grad = new GradTensor(other_grad_shape, other->bidx_, out->shape().size());
  Tensor* this_ptr = this; 
  Tensor* other_ptr = other;

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] { 
    for (std::vector<size_t> l_idx : generate_all_indices(other_ptr->shape())) {
      // update gradient of scalar 
      (this->grad)->at(concat(l_idx, {0})) = 1.0; 
      for (std::vector<size_t> r_idx : generate_all_indices(other_ptr->shape())) {
        std::vector<size_t> idx = concat(l_idx, r_idx);
        if (l_idx == r_idx) {
          (other_ptr->grad)->at(idx) = 1.0;
        }
        else {
          (other_ptr->grad)->at(idx) = 0.0;
        }
      }
    }
  };
  return out;
}

GradTensor* ScalarTensor::add(GradTensor* other) {
  GradTensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] += this->item();
  }
  return out; 
}

ScalarTensor* ScalarTensor::add(ScalarTensor* other) {
  ScalarTensor* out = new ScalarTensor(this->item() + other->item()); 
  this->grad = new GradTensor({1, 1}, 0, 1);
  other->grad = new GradTensor({1, 1}, 0, 1); 

  ScalarTensor* this_ptr = this;  
  ScalarTensor* other_ptr = other;  
  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] {
    (other_ptr->grad)->at({0}) = 1.0; 
    (this->grad)->at({0}) = 1.0; 
  };

  return out; 
}

ScalarTensor* ScalarTensor::add(double* other) {
  ScalarTensor* scalar = new ScalarTensor(this->item() + *other); 
  return this->add(scalar);
}

Tensor* ScalarTensor::sub(Tensor* other) {
  Tensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] = this->item() - out->storage_[i];
  }

  std::vector<size_t> this_grad_shape
    = concat(out->shape(), this->shape()); 
  std::vector<size_t> other_grad_shape
    = concat(out->shape(), other->shape()); 


  this->grad = new GradTensor(this_grad_shape, other->bidx_, out->shape().size());
  other->grad = new GradTensor(other_grad_shape, other->bidx_, out->shape().size());
  Tensor* this_ptr = this; 
  Tensor* other_ptr = other;

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] { 
    for (std::vector<size_t> l_idx : generate_all_indices(other_ptr->shape())) {
      // update gradient of scalar 
      (this->grad)->at(concat(l_idx, {0})) = 1.0; 
      for (std::vector<size_t> r_idx : generate_all_indices(other_ptr->shape())) {
        std::vector<size_t> idx = concat(l_idx, r_idx);
        if (l_idx == r_idx) {
          (other_ptr->grad)->at(idx) = -1.0;
        }
        else {
          (other_ptr->grad)->at(idx) = 0.0;
        }
      }
    }
  };
  return out;
}

GradTensor* ScalarTensor::sub(GradTensor* other) {
  GradTensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] = this->item() - other->storage_[i]; 
  }
  return out; 
}

ScalarTensor* ScalarTensor::sub(ScalarTensor* other) {
  ScalarTensor* out = new ScalarTensor(this->item() - other->item()); 
  this->grad = new GradTensor({1, 1}, 0, 1);
  other->grad = new GradTensor({1, 1}, 0, 1); 

  ScalarTensor* this_ptr = this;  
  ScalarTensor* other_ptr = other;  
  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] {
    (other_ptr->grad)->at({0}) = -1.0; 
    (this->grad)->at({0}) = 1.0; 
  };

  return out; 
}

ScalarTensor* ScalarTensor::sub(double* other) {
  ScalarTensor* scalar = new ScalarTensor(this->item() - *other); 
  return this->sub(scalar);
}

Tensor* ScalarTensor::mul(Tensor* other) {
  Tensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] *= this->item();
  }

  std::vector<size_t> this_grad_shape
    = concat(out->shape(), this->shape()); 
  std::vector<size_t> other_grad_shape
    = concat(out->shape(), other->shape()); 


  this->grad = new GradTensor(this_grad_shape, other->bidx_, out->shape().size());
  other->grad = new GradTensor(other_grad_shape, other->bidx_, out->shape().size());
  Tensor* this_ptr = this; 
  Tensor* other_ptr = other;

  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] { 
    for (std::vector<size_t> l_idx : generate_all_indices(other_ptr->shape())) {
      // update gradient of scalar 
      (this->grad)->at(concat(l_idx, {0})) = other_ptr->at(l_idx); 
      for (std::vector<size_t> r_idx : generate_all_indices(other_ptr->shape())) {
        std::vector<size_t> idx = concat(l_idx, r_idx);
        if (l_idx == r_idx) {
          (other_ptr->grad)->at(idx) = this->item();
        }
        else {
          (other_ptr->grad)->at(idx) = 0.0;
        }
      }
    }
  };
  return out;
}

GradTensor* ScalarTensor::mul(GradTensor* other) {
  GradTensor* out = other->copy();
  size_t length = shape_to_length(other->shape_);
  for (size_t i = 0; i < length; i++) {
    out->storage_[i] *= this->item();
  }
  return out; 
}

ScalarTensor* ScalarTensor::mul(ScalarTensor* other) {
  ScalarTensor* out = new ScalarTensor(this->item() * other->item()); 
  this->grad = new GradTensor({1, 1}, 0, 1);
  other->grad = new GradTensor({1, 1}, 0, 1); 

  ScalarTensor* this_ptr = this;  
  ScalarTensor* other_ptr = other;  
  out->prev = {this_ptr, other_ptr};

  out->backward = [this, other_ptr] {
    (other_ptr->grad)->at({0}) = this->item(); 
    (this->grad)->at({0}) = other_ptr->item(); 
  };
  return out; 
}

ScalarTensor* ScalarTensor::mul(double* other) {
  ScalarTensor* scalar = new ScalarTensor(this->item() * *other);
  return this->mul(scalar);
}

