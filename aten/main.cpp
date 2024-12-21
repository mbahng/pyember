// including src/Tensor.h is essentially a forward declaration of 
// everything inside Tensor.h
#include "./src/Tensor.h" // double quotes mean look in current directory
#include <vector>

int main() {
  Tensor* A = Tensor::gaussian({3, 2}, 0, 1, 0, true);
  Tensor* B = Tensor::gaussian({2, 3}, 0, 1, 0, true); 
  Tensor* C = A->matmul(B); 
  C->backprop(true); 
  return 0;
}
