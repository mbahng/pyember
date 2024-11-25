#include "../Tensor.h" 
#include <vector> 
#include <cassert>

std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape);
std::vector<size_t> duplicate_indices(const std::vector<size_t> shape);

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
  std::vector<size_t> pairshape = duplicate_indices(this->shape_);
  this->grad = GradTensor(pairshape, (this->shape_).size()); 

  // initialize grad[i, i] to 1s, where i may be a vector  
  for (std::vector<size_t> i : generate_all_indices(this->shape_)) { 
    std::vector<size_t> i_dup = duplicate_indices(i); 
    (this->grad).at(i_dup) = 1.0; 
  } 

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



