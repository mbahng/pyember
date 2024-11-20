#include "../Tensor.h"

Tensor Tensor::copy() const {
  return Tensor(this->storage_, this->shape_);
}

