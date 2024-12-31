#include "../Tensor.h"
#include "../../Util/utils.h"


Tensor* Tensor::shallowcopy(bool requires_grad) const {
  // creates a shallow copy 
  return new Tensor(this->_storage, this->_shape, this->bidx, requires_grad); 
}

Tensor* Tensor::deepcopy(bool requires_grad) const {
  // creates a deep copy 
  std::vector<double> storage = this->_storage; 
  std::vector<size_t> shape = this->_shape;  
  size_t bidx = this->bidx; 
  return new Tensor(storage, shape, bidx, requires_grad); 
}

Tensor* Tensor::copy(bool requires_grad) const {
  // alias for shallow copy
  return this->shallowcopy(requires_grad);
}

Tensor* Tensor::reshape(std::vector<size_t> new_shape, bool inplace, bool requires_grad) { 
  if (inplace) {
    this->_shape = new_shape; 
    return this; 
  }
  else {
    Tensor* out = new Tensor(_storage, new_shape, requires_grad);
    return out; 
  }
}

Tensor* Tensor::transpose(size_t d1, size_t d2, bool requires_grad) {
  if (d1 >= this->rank() || d2 >= this->rank()) {
    throw std::invalid_argument("Transposed ranks are out of bounds.");
  }

  if (d1 < this->bidx || d2 < this->bidx) {
    // this should be an error
    throw std::logic_error("You are attempting to transpose the batch ranks. The result bidx will be reset to 0.");
  }
  
  // newshape is divided into 5 parts: before d1, d1, between, d2, after d2 
  std::vector<size_t> before = std::vector<size_t>(this->shape().begin(), this->shape().begin() + d1);  
  std::vector<size_t> d1_idx = std::vector<size_t>{this->shape()[d1]};
  std::vector<size_t> between = std::vector<size_t>(this->shape().begin() + d1 + 1, this->shape().begin() + d2); 
  std::vector<size_t> d2_idx = std::vector<size_t>{this->shape()[d2]};
  std::vector<size_t> after = std::vector<size_t>(this->shape().begin() + d2 + 1, this->shape().end());

  Tensor* res = new Tensor(
    Index::concat(before, d2_idx, between, d1_idx, after), 
    this->bidx, requires_grad
  );

  for (auto bef : Index::generate_all_indices(before)) {
    for (auto _d1 : Index::generate_all_indices(d1_idx)) {
      for (auto bet : Index::generate_all_indices(between)) {
        for (auto _d2 : Index::generate_all_indices(d2_idx)) {
          for (auto aft : Index::generate_all_indices(after)) {
            res->at(Index::concat(bef, _d2, bet, _d1, aft)) = this->at(Index::concat(bef, _d1, bet, _d2, aft));
          }
        }
      }
    }
  }
  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, res, d1, d2] {
    if (this_ptr->requires_grad) {
      this_ptr->grad = new GradTensor(
        Index::concat(this_ptr->bshape(), res->nbshape(), this_ptr->nbshape()), 
        this_ptr->bidx, 
        this_ptr->rank() 
      ); 

      std::vector<size_t> before = std::vector<size_t>(this_ptr->shape().begin() + this_ptr->bidx, this_ptr->shape().begin() + d1);  
      std::vector<size_t> d1_idx = std::vector<size_t>{this_ptr->shape()[d1]};
      std::vector<size_t> between = std::vector<size_t>(this_ptr->shape().begin() + d1 + 1, this_ptr->shape().begin() + d2); 
      std::vector<size_t> d2_idx = std::vector<size_t>{this_ptr->shape()[d2]};
      std::vector<size_t> after = std::vector<size_t>(this_ptr->shape().begin() + d2 + 1, this_ptr->shape().end());

      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto bef : Index::generate_all_indices(before)) {
          for (auto _d1 : Index::generate_all_indices(d1_idx)) {
            for (auto bet : Index::generate_all_indices(between)) {
              for (auto _d2 : Index::generate_all_indices(d2_idx)) {
                for (auto aft : Index::generate_all_indices(after)) {
                  this_ptr->grad->at(Index::concat(b, 
                    bef, _d2, bet, _d1, aft, 
                    bef, _d1, bet, _d2, aft
                    )) = 1.0;
                }
              }
            }
          }
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::transpose(bool requires_grad) {
  return this->transpose(this->rank() - 2, this->rank() - 1);  
}

Tensor* Tensor::transpose(const std::vector<size_t>& axes, bool requires_grad) {
  throw std::logic_error("Not implemented.");
}

Tensor* Tensor::squeeze(bool inplace, bool requires_grad) {
  std::vector<size_t> newshape; 
  for (auto s : this->shape()) {
    if (s != 1) {
      newshape.push_back(s);
    }
  }
  if (newshape.size() == 0) {
    newshape.push_back(1);
  } 
  return this->reshape(newshape, inplace, requires_grad);
}

Tensor* Tensor::squeeze(size_t dim, bool inplace, bool requires_grad) { 
  if (this->_shape[dim] != 1) {
    throw std::logic_error("The dimension you are squeezing is not 1.");
  }
  std::vector<size_t> newshape; 
  for (int i = 0; i < this->hdim(); i++) { 
    if (i != dim) {
      newshape.push_back(this->shape()[i]);
    }
  }
  return this->reshape(newshape, inplace, requires_grad); 
}

Tensor* Tensor::unsqueeze(size_t dim, bool inplace, bool requires_grad) { 
  // dim = dimension that you want to add the 1 in 
  std::vector<size_t> newshape; 
  for (int i = 0; i < this->hdim(); i++) {
    if (i == dim) {
      newshape.push_back(1); 
    }
    newshape.push_back(this->shape()[i]); 
  }
  return this->reshape(newshape, inplace, requires_grad); 
}

