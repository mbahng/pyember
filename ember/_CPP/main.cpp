#include <iostream>
#include <vector>
#include "tensor.h"

int main() {

  Tensor a = Tensor(std::vector<double>{2, 3});
  Tensor h = a.pow(2);
  Tensor b = Tensor(std::vector<double>{3, 5});
  
  Tensor c = b.mult(h); 
  
  Tensor d = Tensor(std::vector<double>{10, 1}); 
  Tensor e = c.dot(d);
  Tensor f = Tensor(std::vector<double>{-2}); 
  Tensor g = f.add(e); 
  
  std::vector<Tensor*> top_sort = g.backprop(); 

  return 0; 
}
