// including src/Tensor.h is essentially a forward declaration of 
// everything inside Tensor.h
#include "./src/Tensor.h" // double quotes mean look in current directory
#include <vector>

int main() {
  Tensor* a = new Tensor({2}, std::vector<size_t>{1});
  Tensor* b = new Tensor({3}, std::vector<size_t>{1}); 
  Tensor* c = a->mul(b); 
  c->backprop(true);
  return 0;
}
