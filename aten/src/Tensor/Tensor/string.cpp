#include "../Tensor.h"

Tensor::operator std::string() const {
  std::string result = BaseTensor::operator std::string(); 
  std::string hg = this->requires_grad ? "True" : "False";
  if (result.back() != '\n') {
    result += ", requires_grad = " + hg + '\n'; 
  }
  else {
    result.pop_back();
    result += ", requires_grad = " + hg + '\n'; 
  }
  return result; 
}

std::string Tensor::meta() const {
  std::string result = BaseTensor::meta(); 
  result.pop_back();  
  std::string hg = requires_grad ? "True" : "False";
  result += ", requires_grad = " + hg + '\n';  

  return result;
}


