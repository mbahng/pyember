#include <vector>
#include <cassert>
#include "../Tensor.h" 
#include "../../Util/utils.h"

// Scalar Operations
GradTensor* GradTensor::operator+(double other) {
  GradTensor* res = this->copy(); 
  for (int bi = 0; bi < (this->_storage).size(); ++bi) {
    res->_storage[bi] += other;
  }
  return res;
}

GradTensor* GradTensor::operator-(double other) {
  GradTensor* res = this->copy(); 
  for (int bi = 0; bi < (this->_storage).size(); ++bi) {
    res->_storage[bi] -= other;
  }
  return res;
}

GradTensor* GradTensor::operator*(double other) {
  GradTensor* res = this->copy(); 
  for (int bi = 0; bi < (this->_storage).size(); ++bi) {
    res->_storage[bi] *= other;
  }
  return res;
}

// GradTensor o GradTensor Operations 
// Used for when we want to modify gradients 
// This supports adding with Scalar GradTensors for broadcasting 
GradTensor* GradTensor::operator+(GradTensor* other) { 
  if (this->is_scalar()) {
    return *other + this->item(); 
  }
  else if (other->is_scalar()) {
    return *this + other->item(); 
  }

  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  GradTensor* res = new GradTensor(r.shape, std::max(this->bidx, other->bidx), r.pidx); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 

  if (this->shape().size() >= other->shape().size()) {  
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) + other->at(i); 
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(other->bshape())) {
      for (auto i : Index::generate_all_indices(other->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) + other->at(Index::concat(b, i)); 
      }
    }
  }
  return res;
}

GradTensor* GradTensor::operator-(GradTensor* other) {
  if (this->is_scalar()) {
    return *other - this->item(); 
  }
  else if (other->is_scalar()) {
    return *this - other->item();
  }
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  GradTensor* res = new GradTensor(r.shape, std::max(this->bidx, other->bidx), r.pidx); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {  
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) - other->at(i); 
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(other->bshape())) {
      for (auto i : Index::generate_all_indices(other->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) - other->at(Index::concat(b, i)); 
      }
    }
  }
  return res;
}

GradTensor* GradTensor::operator*(GradTensor* other) {
  if (this->is_scalar()) {
    return *other * this->item(); 
  }
  else if (other->is_scalar()) {
    return *this * other->item(); 
  }

  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  GradTensor* res = new GradTensor(r.shape, std::max(this->bidx, other->bidx), r.pidx); 
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) {  
    for (auto b : Index::generate_all_indices(this->bshape())) {
      for (auto i : Index::generate_all_indices(this->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) * other->at(i); 
      }
    }
  }
  else {
    for (auto b : Index::generate_all_indices(other->bshape())) {
      for (auto i : Index::generate_all_indices(other->nbshape())) {
        res->at(Index::concat(b, i)) = this->at(i) * other->at(Index::concat(b, i)); 
      }
    }
  }
  return res;
}

// GradTensor + Tensor Operations 
// This should be strictest and should not support Scalar broadcasting 
// It does support regular broadcasting for when we want to take a batch of 
// gradients and add it to a tensor
Tensor* GradTensor::operator+(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape);
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) { 
    for (std::vector<size_t> b : Index::generate_all_indices(this->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) + other->at(i);
      }
    }
  }
  else {
    for (std::vector<size_t> b : Index::generate_all_indices(other->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(i) + other->at(Index::concat(b, i));
      }
    }
  }

  return res;
}

Tensor* GradTensor::operator-(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape);
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) { 
    for (std::vector<size_t> b : Index::generate_all_indices(this->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) - other->at(i);
      }
    }
  }
  else {
    for (std::vector<size_t> b : Index::generate_all_indices(other->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(i) - other->at(Index::concat(b, i));
      }
    }
  }

  return res;
}

Tensor* GradTensor::operator*(Tensor* other) {
  OIntegrity::Shape r = OIntegrity::compat(this, other);  
  Tensor* res = new Tensor(r.shape);
  size_t bs = CIntegrity::prod(r.nb_shape); // batch size 
  
  if (this->shape().size() >= other->shape().size()) { 
    for (std::vector<size_t> b : Index::generate_all_indices(this->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(Index::concat(b, i)) * other->at(i);
      }
    }
  }
  else {
    for (std::vector<size_t> b : Index::generate_all_indices(other->bshape())) {
      for (std::vector<size_t> i : Index::generate_all_indices(this->shape())) {
        res->at(Index::concat(b, i)) = this->at(i) * other->at(Index::concat(b, i));
      }
    }
  }

  return res;
}

// Tensor Contraction of this * other for Chain Rule 
// unlike matrix multiplication of regular Tensors, we don't need to worry 
// about batching across multiple dimensions, since the only time you 
// will do grad matmul is when doing tensor contractions. 
// e.g. (2, 3, 4) x (4, 4, 3) will never happen. 
// other's batch indices will always cover this's indices. 
GradTensor* GradTensor::matmul(GradTensor* other) {   
  // this is repetitive, should fix this to be faster
  OIntegrity::Shape r = OIntegrity::matmul_compat(this, other); 
  GradTensor* res = new GradTensor(r.shape, r.b_shape.size(), r.pidx); 

  std::vector<size_t> B = this->bidx > 0 ? this->bshape() : other->bshape(); 
  std::vector<size_t> C1 = std::vector<size_t>(this->shape().begin() + this->bidx, this->shape().begin() + this->pidx()); 
  std::vector<size_t> C2 = std::vector<size_t>(other->shape().begin() + other->bidx, other->shape().begin() + other->pidx());
  std::vector<size_t> C3 = std::vector<size_t>(other->shape().begin() + other->pidx(), other->shape().end()); 

  // switch based on batches

  if (this->bidx == 0 && other->bidx == 0) {
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
  else if (this->bidx > 0 && other->bidx > 0) {
    for (std::vector<size_t> b : Index::generate_all_indices(B)) {
      for (std::vector<size_t> i : Index::generate_all_indices(C1)) {
        for (std::vector<size_t> k : Index::generate_all_indices(C3)) { 
          double contraction = 0.0;
          for (std::vector<size_t> j : Index::generate_all_indices(C2)) {  
            contraction += this->at(Index::concat(b, i, j)) * other->at(Index::concat(b, j, k)); 
          }
          res->at(Index::concat(b, i, k)) = contraction; 
        }
      }
    }
  }
  else if (this->bidx > 0) {
    for (std::vector<size_t> b : Index::generate_all_indices(B)) {
      for (std::vector<size_t> i : Index::generate_all_indices(C1)) {
        for (std::vector<size_t> k : Index::generate_all_indices(C3)) { 
          double contraction = 0.0;
          for (std::vector<size_t> j : Index::generate_all_indices(C2)) {  
            contraction += this->at(Index::concat(b, i, j)) * other->at(Index::concat(j, k)); 
          }
          res->at(Index::concat(b, i, k)) = contraction; 
        }
      }
    }
  }
  else {
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

  return res; 
}
