#include "../Tensor.h"
#include <cassert>
#include "../../Util/utils.h"

GradTensor* GradTensor::batchsum() {
  GradTensor* res = new GradTensor(this->nbshape(), 0, this->pidx() - this->_bidx); 

  for (std::vector<size_t> i : Index::generate_all_indices(this->nbshape())) {
    double sum = 0.0; 
    for (std::vector<size_t> b : Index::generate_all_indices(this->bshape())) {
      sum += this->at(Index::concat(b, i));
    }
    res->at(i) = sum; 
  }

  return res; 
}
