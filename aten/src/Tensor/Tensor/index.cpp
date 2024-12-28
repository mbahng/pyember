#include "../Tensor.h"

double Tensor::at(const std::vector<size_t>& indices) const {
  return BaseTensor::at(indices);
}

double& Tensor::at(const std::vector<size_t>& indices) {
  return BaseTensor::at(indices);
}

std::unique_ptr<BaseTensor> Tensor::slice(const std::vector<Slice>& slices) const {
  auto base_result = BaseTensor::slice(slices);
  return std::make_unique<Tensor>(base_result->storage(), base_result->shape(), bidx, requires_grad);
}

