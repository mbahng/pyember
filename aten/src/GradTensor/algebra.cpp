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

GradTensor GradTensor::add(GradTensor& other) {
  assert(this->shape() == other.shape() && this->pivot_ == other.pivot_); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::add(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] + other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::sub(GradTensor& other) {
  assert(this->shape() == other.shape() && this->pivot_ == other.pivot_); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::sub(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] - other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::mul(GradTensor& other) {
  assert(this->shape() == other.shape() && this->pivot_ == other.pivot_); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }

  return GradTensor(res_data, this->shape(), this->pivot_);   
}

Tensor GradTensor::mul(Tensor& other) {
  assert(this->shape() == other.shape()); 
  int length = shape_to_length(this->shape());
  std::vector<double> res_data(length, 0.0);  
  for (int i = 0; i < length; i++) {
    res_data[i] = this->data()[i] * other.data()[i];
  }

  return Tensor(res_data, this->shape()); 
}

GradTensor GradTensor::matmul(GradTensor& other) {
  // right matrix multiplication this * other
  std::vector<size_t> otherL = std::vector<size_t> (other.shape().begin(), other.shape().begin() + other.pivot());
  std::vector<size_t> otherR = std::vector<size_t> (other.shape().begin() + other.pivot(), other.shape().end());
  std::vector<size_t> thisL = std::vector<size_t> (this->shape().begin(), this->shape().begin() + this->pivot());
  std::vector<size_t> thisR = std::vector<size_t> (this->shape().begin() + this->pivot(), this->shape().end());

  assert(thisR == otherL);

  size_t m = shape_to_length(thisL);    // Number of rows in result
  size_t n = shape_to_length(otherR);   // Number of columns in result
  size_t k = shape_to_length(thisR);    // Inner dimension for multiplication
  
  // Create result vector initialized with zeros
  std::vector<double> result(m * n, 0.0);
  
  // Perform matrix multiplication
  for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
          double sum = 0.0;
          for (size_t p = 0; p < k; ++p) {
              // Calculate indices for this and other matrices
              size_t this_idx = i * k + p;
              size_t other_idx = p * n + j;
              sum += this->data()[this_idx] * other.data()[other_idx];
          }
          result[i * n + j] = sum;
      }
  }
  
  size_t new_pivot = shape_to_length(thisL);
  std::vector<size_t> new_shape(thisL);
  new_shape.insert(new_shape.end(), otherR.begin(), otherR.end());
  
  return GradTensor(result, new_shape, new_pivot);
}


