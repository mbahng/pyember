#include "../Tensor.h"
#include <vector>
#include <cassert>

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

Tensor* Tensor::sum() { 
  std::vector<double> out_data {0.0};  
  size_t length = shape_to_length(this->shape());
  for (int i = 0; i < length; ++i) {
    out_data[0] += this->data()[i]; 
  }

  Tensor* out = new Tensor(out_data); 
  out->prev = std::vector<Tensor*> {this}; 
 
  std::vector<size_t> newshape = {1}; 
  newshape.insert(newshape.end(), this->shape().begin(), this->shape().end());

  this->grad = new GradTensor(
    std::vector<double>(length, 1.0), 
    newshape,
    1
  ); 

  out->backward = [this, length] {
    for (size_t i = 0; i < length; ++i) {
      (this->grad)->storage_[i] = 1.0;
    }
  };
  return out; 
}

