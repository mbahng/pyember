#include "../Tensor.h"
#include "../../Util/utils.h"
#include <string>

const std::vector<double>& BaseTensor::storage() const { return _storage; }
const size_t BaseTensor::rank() const { return _rank; }
const size_t BaseTensor::hdim() const { return _rank; }
const std::vector<size_t>& BaseTensor::shape() const { return _shape; }
const std::vector<size_t> BaseTensor::bshape() const { return _bshape; }
const std::vector<size_t> BaseTensor::nbshape() const { return _nbshape; }
const size_t& BaseTensor::size() const { return _size; }

bool BaseTensor::is_scalar() const {
  return ((this->shape()).size() == 0) || CIntegrity::prod(this->shape()) == 1; 
}

double BaseTensor::item() const {
  if (!this->is_scalar()) {
    throw std::logic_error("You cannot call item() on a non-scalar tensor.");
  }
  return this->_storage[0];
}

std::string BaseTensor::type() const { return "BaseTensor"; }
std::string BaseTensor::dtype() const { return "double"; }
