#include <vector>
#include <cassert>
#include "../Tensor.h" 

int shape_to_length(std::vector<size_t> shape); 
void print(std::vector<size_t> input);

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

std::vector<size_t> concat_indices(
    std::vector<size_t> shape1,
    std::vector<size_t> shape2);

std::vector<size_t> duplicate_indices(const std::vector<size_t> shape);
std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx);
size_t prod(std::vector<size_t> input); 

GradTensor* GradTensor::add(GradTensor* other) {
  if (this->nb_indices() != other->nb_indices()) {
    throw std::logic_error("Shapes do not match across non-batch dimensions.");
  }
  else if (this->pidx() - this->bidx() != other->pidx() - other->bidx()) {
    throw std::logic_error("Pivots do not match");
  } 

  // set up new shape 
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  std::vector<size_t> new_shape = concat_indices(this->b_indices(), other->shape());  
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
  
  // Create result vector initialized with zeros 
  std::vector res_shape = concat_indices(concat_indices(this->b_indices(), other->b_indices()), concat_indices(thisL, otherR));
  size_t pidx = this->bidx() + other->bidx() + thisL.size(); 
  size_t bidx = this->bidx() + other->bidx();
  GradTensor* out = new GradTensor(res_shape, pidx, bidx); 

  for (std::vector<size_t> b1 : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> b2 : generate_all_indices(other->b_indices())) {
      for (std::vector<size_t> m : generate_all_indices(thisL)) {
        for (std::vector<size_t> k : generate_all_indices(otherR)) {
          double contraction = 0.0; 
          for (std::vector<size_t> n : generate_all_indices(thisR)) {
            std::vector<size_t> l_idx = concat_indices(m, n);
            std::vector<size_t> r_idx = concat_indices(n, k); 
            contraction += (this->at(concat_indices(b1, l_idx)) * other->at(concat_indices(b2, r_idx)));
          }
          std::vector<size_t> batch_idx = concat_indices(b1, b2); 
          out->at(concat_indices(batch_idx, concat_indices(m, k))) = contraction; 
        }
      }
    }
  }

  return out; 
}
