#include "../Tensor.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm> 
#include "../utils.h"

Tensor* Tensor::dot(Tensor* other) { 
  OIntegrity::Shape r = OIntegrity::compat(this, other); 

  Tensor* res = new Tensor(
    Index::concat(r.b_shape, std::vector<size_t>{1}),
    std::max(this->bidx(), other->bidx()), 
    this->requires_grad || other->requires_grad
  ); 

  if (this->shape().size() >= other->shape().size()) {
    for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
      double dot_product = 0.0; 
      for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
        dot_product += this->at(Index::concat(b, i)) * other->at(i); 
      }
      res->at(Index::concat(b, std::vector<size_t>{0})) = dot_product; 
    }
  }
  else {
    for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
      double dot_product = 0.0; 
      for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
        dot_product += this->at(i) * other->at(Index::concat(b, i)); 
      }
      res->at(Index::concat(b, std::vector<size_t>{0})) = dot_product; 
    }
  }

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (other->requires_grad) { res->_prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, other, res, r] { 
    std::vector<size_t> newshape = Index::concat(r.b_shape, std::vector<size_t>{1}, r.nb_shape); 
    size_t bidx = std::max(this_ptr->_bidx, other->_bidx); 
    size_t pidx = bidx + 1 ; 

    if (this_ptr->requires_grad) {
      this_ptr->grad = new GradTensor(newshape, bidx, pidx);  
      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(Index::concat(b, std::vector<size_t>{0}, i)) = other->at(i);
          }
        }
      }
      else {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(Index::concat(b, std::vector<size_t>{0}, i)) = other->at(Index::concat(b, i));
          }
        }
      }
    }

    if (other->requires_grad) {
      other->grad = new GradTensor(newshape, bidx, pidx); 
      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (other->grad)->at(Index::concat(b, std::vector<size_t>{0}, i)) = this_ptr->at(Index::concat(b, i));
          }
        }
      }
      else{
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (other->grad)->at(Index::concat(b, std::vector<size_t>{0}, i)) = this_ptr->at(i);
          }
        }
      }
    }
  };
  return res; 
}

Tensor* Tensor::sum() { 
  double out_data = 0.0;
  size_t length = CIntegrity::prod(this->shape());
  for (int i = 0; i < length; ++i) {
    out_data += this->storage()[i]; 
  }

  Tensor* out = new Tensor(
    {out_data}, 
    std::vector<size_t>{1}, 
    0, 
    this->requires_grad
  ); 

  out->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { out->_prev.push_back(this); } 

  Tensor* this_ptr = this; 
 
  out->_backward = [this_ptr] { 
    if (this_ptr->requires_grad) {
      std::vector<size_t> newshape = Index::concat(this_ptr->bshape(), {1}, this_ptr->nbshape());
      this_ptr->grad = new GradTensor(newshape, this_ptr->_bidx, this_ptr->_bidx + 1);  
      for (auto i : Index::generate_all_indices(this_ptr->grad->shape())) {
        this_ptr->grad->at(i) = 1.0;
      }
    }
  };
  return out; 
}

Tensor* Tensor::sum(size_t dim) {
  if (dim < this->hdim() || dim < 0) {
    throw std::logic_error("The dimensions you are summing up on is bad.");
  }
} 

Tensor* Tensor::sum(std::vector<size_t> dims) {

} 

Tensor* Tensor::pow(double* x) { 
  Tensor* out = this->copy(); 

  for (int i = 0; i < (this->_storage).size(); i++) {
    (out->_storage)[i] = std::pow((this->_storage)[i], *x);
  }

  out->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { out->_prev.push_back(this); }

  // need to allocate this x on heap for it to be accessible by _backward
  double* x_ptr = new double(2); 
  
  Tensor* this_ptr = this; 
  out->_backward = [this_ptr, x_ptr] {  
    if (this_ptr->requires_grad) {
      std::vector<size_t> newshape = Index::concat(this_ptr->shape(), this_ptr->nbshape());
      this_ptr->grad = new GradTensor(newshape, this_ptr->_bidx, (this_ptr->shape()).size());  

      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = *x_ptr * std::pow(this_ptr->at(Index::concat(b, i)), (*x_ptr)-1);  
        }
      }
    }
  };
  return out; 
}

Tensor* Tensor::relu() { 
  Tensor* out = this->copy(); 

  for (int i = 0; i < (this->_storage).size(); i++) {
    if ((this->_storage)[i] < 0.0) { 
      (out->_storage)[i] = 0.0;
    }
  }

  out->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { out->_prev.push_back(this); }

  Tensor* this_ptr = this;  

  out->_backward = [this_ptr] { 
    if (this_ptr->requires_grad) {
      std::vector<size_t> newshape = Index::concat(
        this_ptr->bshape(), 
        this_ptr->nbshape(), 
        this_ptr->nbshape()
      );
      this_ptr->grad = new GradTensor(newshape, this_ptr->bidx(), (this_ptr->shape()).size());  

      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          if (this_ptr->at(Index::concat(b, i)) >= 0.0) {
            (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
          }
        }
      }
    }
  };
  return out; 
}


