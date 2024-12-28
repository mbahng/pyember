#include "../Tensor.h"

// Getter methods
size_t GradTensor::pidx() const { return _pidx; }
std::string GradTensor::type() const { return "GradTensor"; }
std::string GradTensor::dtype() const { return "double"; }

