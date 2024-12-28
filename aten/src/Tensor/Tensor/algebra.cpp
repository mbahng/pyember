#include <vector>
#include <cassert>
#include "../Tensor.h"
#include "../../Util/utils.h"

Tensor* Tensor::add(double other) {
  // initialize scalar tensor so we can work with gradients
  Tensor* scalar = new Tensor(other);
  Tensor* res = this->copy(); 

  for (int bi = 0; bi < this->_storage.size(); ++bi) {
    res->_storage[bi] += other;
  } 

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (scalar->requires_grad) { res->_prev.push_back(scalar); }
  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, scalar, res] {
    if (this_ptr->requires_grad) {
      this_ptr-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), this_ptr->nbshape()), 
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (scalar->requires_grad) {
      scalar-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), std::vector<size_t>{1}),
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, std::vector<size_t>{0})) = 1.0; 
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::sub(double other) {
  // initialize scalar tensor so we can work with gradients
  Tensor* scalar = new Tensor(other);
  Tensor* res = this->copy(); 

  for (int bi = 0; bi < this->_storage.size(); ++bi) {
    res->_storage[bi] -= other;
  } 

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (scalar->requires_grad) { res->_prev.push_back(scalar); }
  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, scalar, res] {
    if (this_ptr->requires_grad) {
      this_ptr-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), this_ptr->nbshape()), 
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (scalar->requires_grad) {
      scalar-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), std::vector<size_t>{1}),
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, std::vector<size_t>{0})) = -1.0; 
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::mul(double other) {
  // initialize scalar tensor so we can work with gradients
  Tensor* scalar = new Tensor(other);
  Tensor* res = this->copy(); 

  for (int bi = 0; bi < this->_storage.size(); ++bi) {
    res->_storage[bi] *= other;
  } 

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (scalar->requires_grad) { res->_prev.push_back(scalar); }
  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, scalar, res] {
    if (this_ptr->requires_grad) {
      this_ptr-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), this_ptr->nbshape()), 
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = scalar->item(); 
        }
      }
    }
    if (scalar->requires_grad) {
      scalar-> grad = new GradTensor(
        Index::concat(this_ptr->bshape(), this_ptr->nbshape(), std::vector<size_t>{1}),
        this_ptr->bidx, 
        this_ptr->shape().size()
      );
      for (auto b : Index::generate_all_indices(this_ptr->bshape())) {
        for (auto i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(
            Index::concat(b, i, std::vector<size_t>{0})
          ) = this_ptr->at(Index::concat(b, i));
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::add(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = CIntegrity::prod(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->storage()[i] + other->storage()[i];
  }

  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::sub(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = CIntegrity::prod(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->storage()[i] - other->storage()[i];
  }
  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::mul(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = CIntegrity::prod(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->storage()[i] * other->storage()[i];
  }
  return new Tensor(res_data, this->shape());   
}

Tensor* Tensor::iadd(GradTensor* other) {
  if (this->bidx < other->bidx) {
    throw std::logic_error("You cannot iadd a tensor with a larger batch shape to this.");
  }
  size_t bs = CIntegrity::prod(this->bshape()); // batch size 

  for (auto b : Index::generate_all_indices(this->bshape())) {
    for (auto i : Index::generate_all_indices(this->nbshape())) {
      this->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) + other->at(i);
    }
  }

  return this; 
}

Tensor* Tensor::isub(GradTensor* other) {
  if (this->bidx < other->bidx) {
    throw std::logic_error("You cannot iadd a tensor with a larger batch shape to this.");
  }
  size_t bs = CIntegrity::prod(this->bshape()); // batch size 

  for (auto b : Index::generate_all_indices(this->bshape())) {
    for (auto i : Index::generate_all_indices(this->nbshape())) {
      this->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) - other->at(i);
    }
  }

  return this; 
}

Tensor* Tensor::imul(GradTensor* other) {
  if (this->bidx < other->bidx) {
    throw std::logic_error("You cannot iadd a tensor with a larger batch shape to this.");
  }
  size_t bs = CIntegrity::prod(this->bshape()); // batch size 

  for (auto b : Index::generate_all_indices(this->bshape())) {
    for (auto i : Index::generate_all_indices(this->nbshape())) {
      this->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) * other->at(i);
    }
  }

  return this; 
}

Tensor* Tensor::add(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(
    r.shape, 
    std::max(this->bidx, other->bidx), 
    this->requires_grad || other->requires_grad
  ); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) + other->at(i);
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) + other->at(Index::concat(b, i));
      }
    }
  }

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (other->requires_grad) { res->_prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, other, res, r] { 
    std::vector<size_t> newshape = Index::concat(r.b_shape, r.nb_shape, r.nb_shape);
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->requires_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
        for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (other->requires_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
        for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
          (other->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::sub(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(
    r.shape, 
    std::max(this->bidx, other->bidx), 
    this->requires_grad || other->requires_grad
  ); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) - other->at(i);
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) - other->at(Index::concat(b, i));
      }
    }
  }

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (other->requires_grad) { res->_prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, other, res, r] {  
    std::vector<size_t> newshape = Index::concat(this_ptr->bshape(), this_ptr->nbshape(), this_ptr->nbshape()); 
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->requires_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (other->requires_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (other->grad)->at(Index::concat(b, i, i)) = -1.0; 
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::mul(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(
    r.shape, 
    std::max(this->bidx, other->bidx), 
    this->requires_grad || other->requires_grad
  ); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) * other->at(i);
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) * other->at(Index::concat(b, i));
      }
    }
  }

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (other->requires_grad) { res->_prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, other, res, r] { 
    std::vector<size_t> newshape = Index::concat(r.b_shape, r.nb_shape, r.nb_shape);
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->requires_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 

      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(Index::concat(b, i, i)) = other->at(i); 
          }
        }
      } 
      else {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (this_ptr->grad)->at(Index::concat(b, i, i)) = other->at(Index::concat(b, i)); 
          }
        }
      }
    }
    if (other->requires_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx, other->bidx), pidx); 

      if (this_ptr->shape().size() >= other->shape().size()) {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (other->grad)->at(Index::concat(b, i, i)) = this_ptr->at(Index::concat(b, i)); 
          }
        }
      }
      else {
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
            (other->grad)->at(Index::concat(b, i, i)) = this_ptr->at(i); 
          }
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::matmul(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::matmul_compat(this, other); 
  Tensor* res = new Tensor(
    r.shape,
    std::max(this->bidx, other->bidx), 
    this->requires_grad || other->requires_grad
  );

  if (this->shape().size() >= other->shape().size()) { 
    // batch not needed here since we can account for b in i
    std::vector<size_t> C1 = std::vector<size_t>(this->shape().begin(), this->shape().begin() + r.pidx); 
    std::vector<size_t> C2 = std::vector<size_t>(other->shape().begin(), other->shape().begin() + 1);
    std::vector<size_t> C3 = std::vector<size_t>(other->shape().begin() + 1, other->shape().end()); 

    for (std::vector<size_t> i : Index::generate_all_indices(C1)) {
      for (std::vector<size_t> k : Index::generate_all_indices(C3)) {
        double contraction = 0.0;
        for (std::vector<size_t> j : Index::generate_all_indices(C2)) { 
          contraction += this->at(Index::concat(i, j)) * other->at(Index::concat(j, k));
        }
        res->at(Index::concat(i, k)) = contraction; 
      }
    }
  }
  else {
    std::vector<size_t> B = std::vector<size_t>(other->shape().begin(), other->shape().begin() + r.bidx); 
    std::vector<size_t> C1 = std::vector<size_t>(this->shape().begin(), this->shape().begin() + 1); 
    std::vector<size_t> C2 = std::vector<size_t>(this->shape().begin() + 1, this->shape().end());
    std::vector<size_t> C3 = std::vector<size_t>(other->shape().begin() + r.bidx + 1, other->shape().end()); 

    for (std::vector<size_t> b : Index::generate_all_indices(B)) {
      for (std::vector<size_t> i : Index::generate_all_indices(C1)) {
        for (std::vector<size_t> k : Index::generate_all_indices(C3)) {
          double contraction = 0.0;
          for (std::vector<size_t> j : Index::generate_all_indices(C2)) { 
            contraction += this->at(Index::concat(i, j)) * other->at(Index::concat(b, j, k));
          }
          res->at(Index::concat(b, i, k)) = contraction; 
        }
      }
    }
  }

  res->_prev = std::vector<Tensor*> {}; 
  if (this->requires_grad) { res->_prev.push_back(this); }
  if (other->requires_grad) { res->_prev.push_back(other); }

  Tensor* this_ptr = this; 

  res->_backward = [this_ptr, other, r] { 

    std::vector<size_t> M = {this_ptr->shape()[this_ptr->shape().size()-2]};
    std::vector<size_t> N = {this_ptr->shape()[this_ptr->shape().size()-1]};
    std::vector<size_t> P = {other->shape()[other->shape().size()-1]}; 


    if (this_ptr->requires_grad) {
      if (this_ptr->shape().size() >= other->shape().size()) {
        this_ptr->grad = new GradTensor(
          Index::concat(
            r.shape, 
            std::vector<size_t>(this_ptr->shape().end() - 2, this_ptr->shape().end())
          ), 
          std::max(this_ptr->bidx, other->bidx), 
          r.bidx + 2
        ); 
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(M)) {
            for (std::vector<size_t> j : Index::generate_all_indices(P)) {
              for (std::vector<size_t> k : Index::generate_all_indices(N)) {
                (this_ptr->grad)->at(Index::concat(b, i, j, i, k)) = other->at(Index::concat(k, j));
              }
            }
          }
        }
      }
      else {
        this_ptr->grad = new GradTensor(
          Index::concat(r.shape, this_ptr->shape()), 
          std::max(this_ptr->bidx, other->bidx), 
          r.bidx + 2
        );
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(M)) {
            for (std::vector<size_t> j : Index::generate_all_indices(P)) {
              for (std::vector<size_t> k : Index::generate_all_indices(N)) {
                (this_ptr->grad)->at(Index::concat(b, i, j, i, k)) = other->at(Index::concat(b, k, j));
              }
            }
          }
        }

      }
    }

    if (other->requires_grad) {

      if (this_ptr->shape().size() >= other->shape().size()) { 
        other->grad = new GradTensor(
          Index::concat(r.shape, other->shape()),
          std::max(this_ptr->bidx, other->bidx), 
          r.bidx + 2
        ); 
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(M)) {
            for (std::vector<size_t> j : Index::generate_all_indices(P)) {
              for (std::vector<size_t> k : Index::generate_all_indices(N)) { 
                (other->grad)->at(Index::concat(b, i, j, k, j)) = this_ptr->at(Index::concat(b, i, k));
              }
            }
          }
        }
      }
      else {
        other->grad = new GradTensor(
          Index::concat(
            r.shape, 
            std::vector<size_t>(other->shape().end() - 2, other->shape().end())
          ),
          std::max(this_ptr->bidx, other->bidx), 
          r.bidx + 2
        ); 
        for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
          for (std::vector<size_t> i : Index::generate_all_indices(M)) {
            for (std::vector<size_t> j : Index::generate_all_indices(P)) {
              for (std::vector<size_t> k : Index::generate_all_indices(N)) {
                (other->grad)->at(Index::concat(b, i, j, k, j)) = this_ptr->at(Index::concat(i, k));
              }
            }
          }
        }
      }
    }
  };

  return res;
}


