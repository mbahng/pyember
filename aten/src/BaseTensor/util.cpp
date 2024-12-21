#include "../Tensor.h"
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cxxabi.h>

BaseTensor::operator std::string() const {  
  // recursive call on each row. 
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3); 
  oss << this->type() << "\n";

  if (shape_.size() == 1) {
    // Special case for scalar/1D results
    for (int i = 0; i < storage_.size(); i++) {
      oss << std::setw(2) << std::string(2, ' '); 
      if (storage_[i] > 0) {oss << '+' << storage_[i]; }
      else if (storage_[i] == 0) {oss << ' ' << storage_[i]; }
      else {oss << storage_[i]; }
    }
  }
  else {
    std::function<void(std::vector<double>, std::vector<size_t>)> print;
    print = [&](std::vector<double> storage, std::vector<size_t> shape) {
      if (shape.size() == 0) {
        return;
      }
      else if (shape.size() == 1) {
        for (int i = 0; i < shape[0]; i++) {
          oss << std::setw(2) << std::string(shape_.size() - shape.size() - 1, ' '); 

          // take care of padding 
          if (storage[i] > 0) { oss << '+' << storage[i]; } 
          else if (storage[i] == 0) { oss << ' ' << storage[i]; } 
          else { oss << storage[i]; }
        }
        return; 
      }
      
      // Calculate stride for this dimension
      size_t stride = 1;
      for (size_t i = 1; i < shape.size(); i++) {
        stride *= shape[i];
      }
      
      // Call print recursively with remaining dimensions 
      for (int i = 0; i < shape[0]; i++) {
        // Pass subset of storage array 
        if (shape.size() > 2) {
          oss << std::string(shape_.size() - shape.size(), ' ') << "[\n";
        }
        std::vector<double> subset(storage.begin() + i * stride, 
                                  storage.begin() + (i + 1) * stride);
        print(subset, std::vector<size_t>(shape.begin() + 1, shape.end())); 
        if (shape.size() > 2) {
          oss << std::string(shape_.size() - shape.size(), ' ') << "]";
        }
        oss << "\n"; 
      }
    };
    print(storage_, shape_);
  }
  oss << "\n"; 
  oss << "shape = ("; 
  for (int i = 0; i < shape_.size() - 1; i++) {
    oss << shape_[i] << ", ";
  }
  oss << shape_[shape_.size()-1] << "), dtype = " << this->dtype(); 
  return oss.str();
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

  if (flat_idx >= storage_.size()) {
    throw std::logic_error("Index is out of bounds for shape.") ;
  }
  return flat_idx;
}

void BaseTensor::validate_indices(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds.");
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

bool BaseTensor::operator==(BaseTensor& other) const {
  // First, compare shapes
  if (this->shape_ != other.shape_) {
    return false;
  }

  // Compare each element, considering floating-point precision
  const double epsilon = std::numeric_limits<double>::epsilon();
  for (size_t i = 0; i < this->data().size(); ++i) {
    if (std::abs(this->data()[i] - other.data()[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

bool BaseTensor::operator!=(BaseTensor& other) const {
  return !(*this == other);
}

const std::vector<size_t> BaseTensor::b_indices() const { 
  return std::vector<size_t>((this->shape()).begin(), (this->shape()).begin() + this->bidx()); 
}

const std::vector<size_t> BaseTensor::nb_indices() const { 
  return std::vector<size_t>((this->shape()).begin() + this->bidx(), (this->shape()).end()); 
}

