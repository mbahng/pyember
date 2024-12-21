#include "../Tensor.h" 
#include <vector> 
#include <cassert>
#include "../utils.h"

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
  std::vector<size_t> newshape = concat(this->b_indices(), duplicate(this->nb_indices()));
  this->grad = new GradTensor(newshape, (this->shape()).size(), this->bidx()); 

  // initialize grad[i, i] to 1s, where i may be a vector  
  for (std::vector<size_t> b : generate_all_indices(this->b_indices())) {
    for (std::vector<size_t> i : generate_all_indices(this->nb_indices())) { 
      std::vector<size_t> idx = concat(b, i, i); 
      (this->grad)->at(idx) = 1.0; 
    } 
  }

  // Build the topological ordering
  std::vector<Tensor*> topo;
  std::set<Tensor*> visited;
  build_topo(this, visited, topo);

  // Reverse for correct dependency order
  std::reverse(topo.begin(), topo.end()); 

  // Backpropagate through the computation graph
  for (Tensor* v : topo) {
    if (v->has_grad && v->backward) {
      v->backward();
    }
  }

  if (!intermediate) {  
    // go backwards again, accumulating gradients
    for (Tensor* now : topo) { 
      for (Tensor* p : now->prev) { 
        p->grad = (now->grad)->matmul(p->grad);
      }
    }
  }

  return topo;
}



