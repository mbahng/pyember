#include "../Tensor.h" 
#include "../utils.h" 
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
  std::vector<size_t> pairshape = Index::concat(this->bshape(), this->nbshape(), this->nbshape());
  this->grad = new GradTensor(pairshape, this->bidx_, this->shape().size()); 

  if (!(this->has_grad())) {
    throw std::logic_error("This tensor has has_grad=False.");
  }

  // initialize grad[i, i] to 1s, where i may be a vector  
  for (std::vector<size_t> i : Index::generate_all_indices(this->shape_)) { 
    std::vector<size_t> i_dup = Index::concat(i, i); 
    (this->grad)->at(i_dup) = 1.0; 
  } 

  // Build the topological ordering
  std::vector<Tensor*> topo;
  std::set<Tensor*> visited;
  build_topo(this, visited, topo);

  // Reverse for correct dependency order
  std::reverse(topo.begin(), topo.end()); 

  // Backpropagate through the computation graph
  for (Tensor* v : topo) {
    if (v->has_grad() && v->backward) { 
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



