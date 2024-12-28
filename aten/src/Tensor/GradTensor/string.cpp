#include "../Tensor.h"

GradTensor::operator std::string() const {
  std::string result = BaseTensor::operator std::string();  
  if (result.back() != '\n') {
    result += ", pidx = " + std::to_string(this->pidx()) + '\n'; 
  }
  else {
    result.pop_back();
    result += ", pidx = " + std::to_string(this->pidx()) + '\n'; 
  }
  return result; 
}

std::string GradTensor::meta() const {
  std::string result = BaseTensor::meta(); 
  result.pop_back();  
  std::stringstream oss; 
  oss << ", pidx = " << this->pidx() << "\n";
  return result + oss.str();
}

