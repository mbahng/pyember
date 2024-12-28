#include "../Tensor.h"
#include <sstream>
#include <iostream>
#include <cxxabi.h>

double BaseTensor::at(const std::vector<size_t>& indices) const {
  validate_indices(indices);
  return _storage[get_flat_index(indices)];
}

double& BaseTensor::at(const std::vector<size_t>& indices) {
  validate_indices(indices);
  return _storage[get_flat_index(indices)];
}
std::unique_ptr<BaseTensor> BaseTensor::slice(const std::vector<Slice>& slices) const {
  if (slices.size() != _shape.size()) {
    throw std::invalid_argument("Number of slices must match tensor dimensions");
  }

  std::vector<size_t> new_shape = calculate_slice_shape(slices);
  std::vector<double> new_storage;
  std::vector<size_t> current_indices(_shape.size(), 0);
  
  copy_slice_data(slices, current_indices, 0, new_storage);
  
  auto result = std::make_unique<BaseTensor>();
  result->_shape = std::move(new_shape);
  result->_storage = std::move(new_storage); 
  return result;
}

size_t BaseTensor::get_flat_index(const std::vector<size_t>& indices) const {
  size_t flat_idx = 0;
  size_t multiplier = 1;
  
  for (int i = _shape.size() - 1; i >= 0; --i) {
    flat_idx += indices[i] * multiplier;
    multiplier *= _shape[i];
  } 

  if (flat_idx >= _storage.size()) {
    throw std::logic_error("Index is out of bounds for shape.") ;
  }
  return flat_idx;
}

void BaseTensor::validate_indices(const std::vector<size_t>& indices) const {
  if (indices.size() != _shape.size()) {
    std::ostringstream oss; 
    oss << "Attempting to index ( "; 
    for (size_t i : indices) {
      oss << i << " "; 
    } 
    oss << ") from tensor of shape ("; 
    for (size_t i : _shape) {
      oss << i << " ";
    }
    oss << ").";

    throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= _shape[i]) {
      std::ostringstream oss; 
      oss << "Out of Bounds: Attempting to index ( "; 
      for (size_t i : indices) {
        oss << i << " "; 
      } 
      oss << ") from tensor of shape ("; 
      for (size_t i : _shape) {
        oss << i << " ";
      }
      oss << ").";

      throw std::invalid_argument(oss.str());
    }
  }
}

std::vector<size_t> BaseTensor::calculate_slice_shape(const std::vector<Slice>& slices) const {
  std::vector<size_t> new_shape;
  for (size_t i = 0; i < slices.size(); ++i) {
    size_t dim_size = _shape[i];
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
  
  if (current_dim == _shape.size()) {
    result_storage.push_back(_storage[get_flat_index(current_indices)]);
    return;
  }
  
  size_t start = slices[current_dim].start;
  size_t stop = (slices[current_dim].stop == std::numeric_limits<size_t>::max()) 
        ? _shape[current_dim] : slices[current_dim].stop;
  size_t step = slices[current_dim].step;
  
  for (size_t i = start; i < stop; i += step) {
    current_indices[current_dim] = i;
    copy_slice_data(slices, current_indices, current_dim + 1, result_storage);
  }
}

