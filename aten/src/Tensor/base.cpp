#include "../Tensor.h"

std::string Tensor::type() const { return "Tensor"; }
std::string Tensor::dtype() const { return "double"; }

std::vector<Tensor*> Tensor::prev() const {
  return _prev; 
}
