#include <vector>
#include <cassert>
#include "../Tensor.h"
#include "../utils.h"

Tensor* Tensor::add(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape, std::max(this->bidx_, other->bidx_), true); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) { 
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[b * bs + i] + other->storage_[i]; 
      }
    }
  }
  else {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) {
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[i] + other->storage_[b * bs + i]; 
      }
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
    std::vector<size_t> newshape = Index::concat(r.b_shape, r.nb_shape, r.nb_shape);
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
        for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(r.b_shape)) {
        for (std::vector<size_t> i : Index::generate_all_indices(r.nb_shape)) {
          (other->grad)->at(Index::concat(b, i, i)) = 1.0; 
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
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape, std::max(this->bidx_, other->bidx_), true); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) { 
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[b * bs + i] - other->storage_[i]; 
      }
    }
  }
  else {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) {
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[i] - other->storage_[b * bs + i]; 
      }
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
    std::vector<size_t> newshape = Index::concat(this_ptr->bshape(), this_ptr->nbshape(), this_ptr->nbshape()); 
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (this_ptr->grad)->at(Index::concat(b, i, i)) = 1.0; 
        }
      }
    }
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 
      for (std::vector<size_t> b : Index::generate_all_indices(this_ptr->bshape())) {
        for (std::vector<size_t> i : Index::generate_all_indices(this_ptr->nbshape())) {
          (other->grad)->at(Index::concat(b, i, i)) = -1.0; 
        }
      }
    }
  };

  return res; 
}

Tensor* Tensor::sub(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = CIntegrity::prod(this->shape());
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
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape, std::max(this->bidx_, other->bidx_), true); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) { 
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[b * bs + i] * other->storage_[i]; 
      }
    }
  }
  else {
    for (int b = 0; b < CIntegrity::prod(r.b_shape); b++) {
      for (int i = 0; i < CIntegrity::prod(r.nb_shape); i++) {
        res->storage_[b * bs + i] = this->storage_[i] * other->storage_[b * bs + i]; 
      }
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
    std::vector<size_t> newshape = Index::concat(r.b_shape, r.nb_shape, r.nb_shape);
    size_t pidx = r.b_shape.size() + r.nb_shape.size(); 
    if (this_ptr->has_grad) {
      this_ptr->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 

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
    if (other->has_grad) {
      other->grad = new GradTensor(newshape, std::max(this_ptr->bidx_, other->bidx_), pidx); 

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

Tensor* Tensor::mul(GradTensor* other) {
  if (this->shape() != other->shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = CIntegrity::prod(this->shape());
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
  OIntegrity::Shape r = OIntegrity::matmul_compat(this, other); 
  Tensor* res = new Tensor(r.shape);

  if (this->has_grad || other->has_grad) { res->has_grad = true; }
  else { res->has_grad = false; }

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

  res->prev = std::vector<Tensor*> {}; 
  if (this->has_grad) { res->prev.push_back(this); }
  if (other->has_grad) { res->prev.push_back(other); } 

  Tensor* this_ptr = this; 

  res->backward = [this_ptr, other, r] { 

    std::vector<size_t> M = {this_ptr->shape()[this_ptr->shape().size()-2]};
    std::vector<size_t> N = {this_ptr->shape()[this_ptr->shape().size()-1]};
    std::vector<size_t> P = {other->shape()[other->shape().size()-1]}; 


    if (this_ptr->has_grad) {
      if (this_ptr->shape().size() >= other->shape().size()) {
        this_ptr->grad = new GradTensor(
          Index::concat(
            r.shape, 
            std::vector<size_t>(this_ptr->shape().end() - 2, this_ptr->shape().end())
          ), 
          std::max(this_ptr->bidx_, other->bidx_), 
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
          std::max(this_ptr->bidx_, other->bidx_), 
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

    if (other->has_grad) {

      if (this_ptr->shape().size() >= other->shape().size()) { 
        other->grad = new GradTensor(
          Index::concat(r.shape, other->shape()),
          std::max(this_ptr->bidx_, other->bidx_), 
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
          std::max(this_ptr->bidx_, other->bidx_), 
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


