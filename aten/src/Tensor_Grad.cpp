#include "Tensor.h" 
#include <vector> 
#include <cassert>

int shape_to_length(std::vector<size_t> shape);

GradTensor GradTensor::lmatmul(GradTensor& other) {
  // left matrix multiplication other * this
  std::vector<size_t> otherL = std::vector<size_t> (other.shape().begin(), other.shape().begin() + other.pivot());
  std::vector<size_t> otherR = std::vector<size_t> (other.shape().begin() + other.pivot(), other.shape().end());
  std::vector<size_t> thisL = std::vector<size_t> (this->shape().begin(), this->shape().begin() + this->pivot());
  std::vector<size_t> thisR = std::vector<size_t> (this->shape().begin() + this->pivot(), this->shape().end());

  assert(otherR == thisL);

  size_t m = shape_to_length(otherL);    // Number of rows in result
  size_t n = shape_to_length(thisR);     // Number of columns in result
  size_t k = shape_to_length(otherR);    // Inner dimension for multiplication
  
  // Create result vector initialized with zeros
  std::vector<double> result(m * n, 0.0);
  
  // Perform matrix multiplication
  for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
          double sum = 0.0;
          for (size_t p = 0; p < k; ++p) {
              // Calculate indices for other and this matrices
              size_t other_idx = i * k + p;
              size_t this_idx = p * n + j;
              sum += other.data()[other_idx] * this->data()[this_idx];
          }
          result[i * n + j] = sum;
      }
  }
  
  // Create new shape by combining otherL and thisR
  std::vector<size_t> new_shape(otherL);
  new_shape.insert(new_shape.end(), thisR.begin(), thisR.end());
  
  // Calculate new pivot
  size_t new_pivot = shape_to_length(thisR) + 1;
  
  return GradTensor(result, new_shape, new_pivot);
}

GradTensor GradTensor::rmatmul(GradTensor& other) {
  // right matrix multiplication this * other
  std::vector<size_t> otherL = std::vector<size_t> (other.shape().begin(), other.shape().begin() + other.pivot());
  std::vector<size_t> otherR = std::vector<size_t> (other.shape().begin() + other.pivot(), other.shape().end());
  std::vector<size_t> thisL = std::vector<size_t> (this->shape().begin(), this->shape().begin() + this->pivot());
  std::vector<size_t> thisR = std::vector<size_t> (this->shape().begin() + this->pivot(), this->shape().end());

  std::cout << otherL[0] << "\n"; 
  std::cout << otherR[0] << "\n"; 
  std::cout << thisL[0] << "\n"; 
  std::cout << thisR[0] << "\n"; 

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
  
  size_t new_pivot = shape_to_length(otherR) + 1;
  std::vector<size_t> new_shape(thisL);
  new_shape.insert(new_shape.end(), otherR.begin(), otherR.end());
  
  return GradTensor(result, new_shape, new_pivot);
}
