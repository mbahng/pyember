#include "./src/Tensor/Tensor.h" 
#include "./src/Util/utils.h"
#include <vector> 

int main() {
  GradTensor* t1 = new GradTensor(CIntegrity::range(1, 7, 1), {2, 3}, 0, 1);
  GradTensor* t2 = new GradTensor(CIntegrity::range(1, 13, 1), {2, 2, 3}, 1, 2); 
  *t1 + t2;
  return 0;
}
