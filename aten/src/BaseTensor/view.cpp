#include <vector>
#include <stdexcept>
#include "../Tensor.h"

int shape_to_length(std::vector<size_t> shape);

BaseTensor& BaseTensor::reshape(std::vector<size_t> new_shape) {
  if (shape_to_length(new_shape) != shape_to_length(this->shape_)) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  this->shape_ = new_shape; 
  return *this; 
}

Tensor& Tensor::reshape(std::vector<size_t> new_shape) {
  if (shape_to_length(new_shape) != shape_to_length(this->shape_)) {
    throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
  }
  this->shape_ = new_shape; 
  return *this; 
}
double BaseTensor::at(const std::vector<size_t>& indices) const {
  validate_indices(indices);
  return storage_[get_flat_index(indices)];
}

double& BaseTensor::at(const std::vector<size_t>& indices) {
  validate_indices(indices);
  return storage_[get_flat_index(indices)];
}
std::unique_ptr<BaseTensor> BaseTensor::slice(const std::vector<Slice>& slices) const {
  if (slices.size() != shape_.size()) {
    throw std::invalid_argument("Number of slices must match tensor dimensions");
  }

  std::vector<size_t> new_shape = calculate_slice_shape(slices);
  std::vector<double> new_storage;
  std::vector<size_t> current_indices(shape_.size(), 0);
  
  copy_slice_data(slices, current_indices, 0, new_storage);
  
  auto result = std::make_unique<BaseTensor>();
  result->shape_ = std::move(new_shape);
  result->storage_ = std::move(new_storage);
  return result;
}

size_t BaseTensor::get_flat_index(const std::vector<size_t>& indices) const {
  size_t flat_idx = 0;
  size_t multiplier = 1;
  
  for (int i = shape_.size() - 1; i >= 0; --i) {
    flat_idx += indices[i] * multiplier;
    multiplier *= shape_[i];
  }
  
  return flat_idx;
}

void BaseTensor::validate_indices(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
  }
  
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds");
    }
  }
}

std::vector<size_t> BaseTensor::calculate_slice_shape(const std::vector<Slice>& slices) const {
  std::vector<size_t> new_shape;
  for (size_t i = 0; i < slices.size(); ++i) {
    size_t dim_size = shape_[i];
    size_t start = slices[i].start;
    size_t stop = (slices[i].stop == std::numeric_limits<size_t>::max()) 
          ? dim_size : slices[i].stop;
    size_t step = slices[i].step;
    
    if (start >= dim_size) start = dim_size;
    if (stop > dim_size) stop = dim_size;
    
    size_t slice_size = (stop - start + step - 1) / step;
    new_shape.push_back(slice_size);
  }
  return new_shape;
}

void BaseTensor::copy_slice_data(
  const std::vector<Slice>& slices,
  std::vector<size_t>& current_indices,
  size_t current_dim,
  std::vector<double>& result_storage) const {
  
  if (current_dim == shape_.size()) {
    result_storage.push_back(storage_[get_flat_index(current_indices)]);
    return;
  }
  
  size_t start = slices[current_dim].start;
  size_t stop = (slices[current_dim].stop == std::numeric_limits<size_t>::max()) 
        ? shape_[current_dim] : slices[current_dim].stop;
  size_t step = slices[current_dim].step;
  
  for (size_t i = start; i < stop; i += step) {
    current_indices[current_dim] = i;
    copy_slice_data(slices, current_indices, current_dim + 1, result_storage);
  }
}
