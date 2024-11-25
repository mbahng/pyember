#include <vector>
#include <cassert>
#include "../Tensor.h" 

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

std::vector<size_t> concat_indices(
    std::vector<size_t> shape1,
    std::vector<size_t> shape2);

std::vector<size_t> duplicate_indices(const std::vector<size_t> shape);
std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx);

GradTensor GradTensor::add(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  else if (this->pivot_ != other.pivot_) {
    throw std::logic_error("Pivots do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::add(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::sub(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  else if (this->pivot_ != other.pivot_) {
    throw std::logic_error("Pivots do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::sub(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::mul(GradTensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  else if (this->pivot_ != other.pivot_) {
    throw std::logic_error("Pivots do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::mul(Tensor& other) {
  if (this->shape() != other.shape()) {
    throw std::logic_error("Shapes do not match");
  }
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::matmul(GradTensor& other) { 
  // tensor contraction this * other 
  std::vector<size_t> otherL = std::vector<size_t> (other.shape().begin(), other.shape().begin() + other.pivot());
  std::vector<size_t> otherR = std::vector<size_t> (other.shape().begin() + other.pivot(), other.shape().end());
  std::vector<size_t> thisL = std::vector<size_t> (this->shape().begin(), this->shape().begin() + this->pivot());
  std::vector<size_t> thisR = std::vector<size_t> (this->shape().begin() + this->pivot(), this->shape().end());

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
  
  size_t m = shape_to_length(thisL);    // Number of rows in result
  size_t n = shape_to_length(otherR);   // Number of columns in result
  size_t k = shape_to_length(thisR);    // Inner dimension for multiplication
  
  // Create result vector initialized with zeros 
  GradTensor out = GradTensor(duplicate_indices(thisL), thisL.size());  

  for (std::vector<size_t> mk : generate_all_indices(out.shape())) { 
    std::vector<std::vector<size_t>> indices = split_indices(mk, thisL.size()); 
    std::vector<size_t> m = indices[0]; 
    std::vector<size_t> k = indices[1]; 
    double contraction = 0.0; 
    for (std::vector<size_t> n : generate_all_indices(otherR)) {
      std::vector<size_t> l_idx = concat_indices(m, n);
      std::vector<size_t> r_idx = concat_indices(n, k); 
      contraction += this->at(l_idx) * other.at(r_idx);
    }
    out.at(mk) = contraction; 
  }

  return out; 
}
