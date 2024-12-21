#include "./src/Tensor.h" 

int main() {
  Tensor* t1 = Tensor::arange(0, 10, 1)->reshape({5, 2}, true); 
  Tensor* t2 = Tensor::arange(0, 6, 1)->reshape({2, 3}, true); 
  Tensor* prod = t1->matmul(t2); 
  std::cout << std::string(*prod); 
  return 0;
}
