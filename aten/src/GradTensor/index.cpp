#include "../Tensor.h"

double GradTensor::at(const std::vector<size_t>& indices) const {
    return BaseTensor::at(indices);
}

double& GradTensor::at(const std::vector<size_t>& indices) {
    return BaseTensor::at(indices);
}

