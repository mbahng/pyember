#include "./src/Tensor.h" 

int main() {
  Tensor* t1 = Tensor::arange(0, 5);
  Tensor* t2 = Tensor::arange(0, 5); 
  Tensor* s1 = t1->add(t2);
  Tensor* truth_sum = Tensor::arange(0, 10, 2);
  std::cout << std::string(*t1) << "\n"; 
  std::cout << std::string(*t2) << "\n"; 
  std::cout << std::string(*truth_sum) << "\n"; 
  std::cout << std::string(*s1) << "\n"; 
  return 0;
}
