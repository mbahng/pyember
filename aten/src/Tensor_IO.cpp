#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include "Tensor.h"
#include <cxxabi.h>

BaseTensor::operator std::string() const {  
  // recursive call on each row. 
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3); 
  oss << this->type() << "\n";
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
  oss << "\n"; 
  oss << "shape = ("; 
  for (int i = 0; i < shape_.size() - 1; i++) {
    oss << shape_[i] << ", ";
  }
  oss << shape_[shape_.size()-1] << "), dtype = " << this->dtype() << "\n";
  return oss.str();
}
