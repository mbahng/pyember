#include "./src/Tensor.h"
#include <vector>

int main() {
  Tensor* a = new Tensor({2}, std::vector<size_t>{1});
  Tensor* b = new Tensor({3}, std::vector<size_t>{1}); 
  Tensor* c = a->mul(b); 
  c->backprop(true);
  return 0;
}
