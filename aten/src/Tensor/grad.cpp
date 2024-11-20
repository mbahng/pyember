#include "../Tensor.h" 
#include <vector> 
#include <cassert>


void Tensor::build_topo(Tensor* v, std::set<Tensor*>& visited, std::vector<Tensor*>& topo) {
  if (visited.find(v) == visited.end()) {
    visited.insert(v);
    for (Tensor* child : v->prev) {
      build_topo(child, visited, topo);
    }
    topo.push_back(v);
  }
}

std::vector<Tensor*> Tensor::backprop(bool intermediate) {
  // Set the gradient of the final output (this tensor) to 1.0
  auto self_ref = this; 
  this->grad = GradTensor::eye(this->data().size(), 1);
  
  // Build the topological ordering
  std::vector<Tensor*> topo;
  std::set<Tensor*> visited;
  build_topo(this, visited, topo);

  // Reverse for correct dependency order
  std::reverse(topo.begin(), topo.end()); 
  
  // Backpropagate through the computation graph
  for (Tensor* v : topo) {
    if (v->backward) {
      v->backward();
    }
  }

  if (!intermediate) {
    for (Tensor* now : topo) {
      for (Tensor* p : now->prev) {
        p->grad = (now->grad).matmul(p->grad);
      }
    }
  }

  return topo;
}



