#include <vector>
#include <cassert>
#include "../Tensor.h" 
#include "../utils.h" 

GradTensor* GradTensor::add(GradTensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  else if (this->pidx() - this->bidx() != other->pidx() - other->bidx()) {
    throw std::logic_error("Pivots do not match");
  } 

  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] + other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new GradTensor(res_data, new_shape, this->pidx() + other->bidx(), this->bidx() + other->bidx());   
}

Tensor* GradTensor::add(Tensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] + other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new Tensor(res_data, new_shape, this->bidx() + other->bidx());   
}

GradTensor* GradTensor::add(ScalarTensor* other) {
  return other->add(this); 
}

GradTensor* GradTensor::add(double* other) {
  ScalarTensor* scalar = new ScalarTensor(*other);
  return scalar->add(this); 
}

GradTensor* GradTensor::sub(GradTensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  else if (this->pidx() - this->bidx() != other->pidx() - other->bidx()) {
    throw std::logic_error("Pivots do not match");
  } 

  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] - other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new GradTensor(res_data, new_shape, this->pidx() + other->bidx(), this->bidx() + other->bidx());   
}

Tensor* GradTensor::sub(Tensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] - other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new Tensor(res_data, new_shape, this->bidx() + other->bidx());   
}

GradTensor* GradTensor::sub(ScalarTensor* other) {
  GradTensor* out = this->copy(); 
  for (int i = 0; i < this->storage_.size(); i++) {
    out->storage_[i] -= other->item();
  }
  return out; 
}

GradTensor* GradTensor::sub(double* other) {
  ScalarTensor* scalar = new ScalarTensor(*other);
  return this->sub(scalar); 
}

GradTensor* GradTensor::mul(GradTensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  else if (this->pidx() - this->bidx() != other->pidx() - other->bidx()) {
    throw std::logic_error("Pivots do not match");
  } 

  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] * other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new GradTensor(res_data, new_shape, this->pidx() + other->bidx(), this->bidx() + other->bidx());   
}

Tensor* GradTensor::mul(Tensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  // set up new shape 
  std::vector<size_t> new_shape = concat(this->b_indices(), other->shape());  
  size_t new_pidx = (this->pidx()) + (other->bidx()); 

  int length = shape_to_length(new_shape); 
  size_t nb_shift = prod(this->nb_indices());
  size_t b2_shift = prod(other->b_indices());
  std::vector<double> res_data(shape_to_length(new_shape), 0.0);  
  for (int b1 = 0; b1 < prod(this->b_indices()); b1++) {
    for (int b2 = 0; b2 < prod(other->b_indices()); b2++) { 
      for (int i = 0; i < prod(this->nb_indices()); i++) { 
        res_data[nb_shift * prod(other->b_indices()) * b1 + nb_shift * b2 + i] = this->data()[nb_shift * b1 + i] * other->data()[nb_shift * b2 + i];
      }
    }
  }

  return new Tensor(res_data, new_shape, this->bidx() + other->bidx());   
}

GradTensor* GradTensor::mul(ScalarTensor* other) {
  return other->mul(this); 
}

GradTensor* GradTensor::mul(double* other) {
  ScalarTensor* scalar = new ScalarTensor(*other);
  return scalar->mul(this); 
}

GradTensor* GradTensor::matmul(GradTensor* other) {   
  // tensor contraction this * other 
  // unlike matrix multiplication of regular Tensors, we don't need to worry 
  // about batching across multiple dimensions, since the only time you 
  // will do grad matmul is when doing tensor contractions. 
  // e.g. (2, 3, 4) x (4, 4, 3) will never happen. 
  // other's batch indices will always cover this.

  std::vector<size_t> thisL = std::vector<size_t> (this->shape().begin() + this->bidx(), this->shape().begin() + this->pidx());
  std::vector<size_t> thisR = std::vector<size_t> (this->shape().begin() + this->pidx(), this->shape().end());
  std::vector<size_t> otherL = std::vector<size_t> (other->shape().begin() + other->bidx(), other->shape().begin() + other->pidx());
  std::vector<size_t> otherR = std::vector<size_t> (other->shape().begin() + other->pidx(), other->shape().end()); 

  // thisR and otherL are hyperdimesions to be contracted 
  if (thisR != otherL) {
    std::ostringstream msg;
    msg << "Dimensions to be contracted are not equal: left (";  
    for (size_t s : thisR) { msg << " " << s; }
    msg << " ), right ( ";
    for (size_t s : otherL) { msg << " " << s; }
    msg << " )";
    throw std::logic_error(msg.str());
  }
  
  GradTensor* out = new GradTensor(other->shape(), other->pidx(), other->bidx()); 

  std::vector<size_t> diff_indices;
  for (size_t i = this->bidx(); i < other->bidx(); i++) {
      diff_indices.push_back(other->b_indices()[i]);
  }

  for (std::vector<size_t> b_outer : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> b_inner : generate_all_indices(diff_indices)) {
      for (std::vector<size_t> m : generate_all_indices(thisL)) {
        for (std::vector<size_t> k : generate_all_indices(otherR)) {
          double contraction = 0.0; 
          for (std::vector<size_t> n : generate_all_indices(thisR)) {
            std::vector<size_t> l_idx = concat(m, n);
            std::vector<size_t> r_idx = concat(n, k); 
            contraction += (this->at(concat(b_outer, l_idx)) * other->at(concat(b_outer, b_inner, r_idx))); 
          }
          out->at(concat(b_outer, b_inner, m, k)) = contraction; 
        }
      }
    }
  }

  return out; 
}
