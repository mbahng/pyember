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

  if (_shape.size() == 1) {
    // Special case for scalar/1D results
    for (int i = 0; i < _storage.size(); i++) {
      oss << std::setw(2) << std::string(2, ' '); 
      if (_storage[i] > 0) {oss << '+' << _storage[i]; }
      else if (_storage[i] == 0) {oss << ' ' << _storage[i]; }
      else {oss << _storage[i]; }
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
          oss << std::setw(2) << std::string(_shape.size() - shape.size() - 1, ' '); 

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
          oss << std::string(_shape.size() - shape.size(), ' ') << "[\n";
        }
        std::vector<double> subset(storage.begin() + i * stride, 
                                  storage.begin() + (i + 1) * stride);
        print(subset, std::vector<size_t>(shape.begin() + 1, shape.end())); 
        if (shape.size() > 2) {
          oss << std::string(_shape.size() - shape.size(), ' ') << "]";
        }
        oss << "\n"; 
      }
    };
    print(_storage, _shape);
  }
  oss << "\n"; 
  oss << "shape = ("; 
  for (int i = 0; i < _shape.size() - 1; i++) {
    oss << _shape[i] << ", ";
  }
  oss << _shape[_shape.size()-1] << "), dtype = " << this->dtype() << ", bidx = " << this->_bidx; 
  return oss.str();
}

std::string BaseTensor::meta() const {
  std::ostringstream oss; 

  oss << "shape = ( "; 
  for (auto p : this->bshape()) {
    oss << p << " ";
  }
  oss << "| "; 
  for (auto p : this->nbshape()) {
    oss << p << " ";
  } 
  oss << ")\n"; 
  
  return oss.str(); 
}

