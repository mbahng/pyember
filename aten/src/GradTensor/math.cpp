#include "../Tensor.h"
#include <vector>
#include <cassert>
#include "../utils.h"

GradTensor* GradTensor::batchsum() {
  GradTensor* res = new GradTensor(this->nbshape(), 0, this->pidx() - this->bidx_); 

  for (std::vector<size_t> i : generate_all_indices(this->nbshape())) {
    double sum = 0.0; 
    for (std::vector<size_t> b : generate_all_indices(this->bshape())) {
      sum += this->at(concat(b, i));
    }
    res->at(i) = sum; 
  }

  return res; 
}
