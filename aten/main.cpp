#include "./src/Tensor.h" 
#include "./src/utils.h" 
#include <vector> 
#include <iostream> 

int main() {
  Tensor* A = new Tensor(range(0, 12, 1), {2, 6}, 1, true); 
  Tensor* B = new Tensor(range(0, 6, 1), {6}, 0, true); 

  Tensor* C = A->dot(B); 

  std::cout << std::string(*C) << std::endl; 

  C->backprop(false);  
  return 0;
}
