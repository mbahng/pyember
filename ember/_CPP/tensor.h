#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath> 
#include <random> 
#include <ctime>
#include <set>

int shape_to_length(std::vector<int> shape); 

std::vector<std::vector<double>> zero_matrix(int n);

std::vector<std::vector<double>> zero_matrix(int n, int m);

std::vector<std::vector<double>> eye_matrix(int n);


class Tensor {
public: 
  std::vector<double> data; 

  // autograd features 
  // really the total derivative, i.e. Jacobian rather than gradient
  std::vector<std::vector<double>> grad; 
  std::vector<Tensor*> prev;
  std::function<void()> backward; 

  Tensor(std::vector<double> data) {
    this->data = data; 
    
    // previous Tensors used to compute this tensor, if any
    this->prev = std::vector<Tensor*>(); 
  }

  void build_topo(Tensor* v, std::set<Tensor*>& visited, std::vector<Tensor*>& topo) {
    if (visited.find(v) == visited.end()) {
      visited.insert(v);
      for (Tensor* child : v->prev) {
        build_topo(child, visited, topo);
      }
      topo.push_back(v);
    }
  }

  std::vector<Tensor*> backprop() {
    // Set the gradient of the final output (this tensor) to 1.0
    this->grad = eye_matrix(this->data.size());
    
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

    return topo;
  }

  operator std::string() const { 
    std::ostringstream oss; 
    oss << std::fixed << std::setprecision(2);
    for (int i = 0; i < data.size(); ++i) {
      oss << std::setw(8) << data[i]; 
    }
    return oss.str(); 
  }

  Tensor add(Tensor& other) {
    // vector addition
    std::vector<double> res_data(data.size(), 0.0); 
    Tensor out = Tensor(res_data); 
    for (int i = 0; i < res_data.size(); ++i) {
      out.data[i] = this->data[i] + other.data[i]; 
    }

    this->grad = zero_matrix(data.size()); 
    other.grad = zero_matrix(data.size()); 

    Tensor* self_ptr = this; 
    Tensor* other_ptr = &other; 

    out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 
  
    out.backward = [self_ptr, other_ptr] {
      self_ptr->grad = eye_matrix(self_ptr->data.size()); 
      other_ptr->grad = eye_matrix(other_ptr->data.size()); 
    };
    return out; 
  }

  Tensor mult(Tensor &other) { 
    // element-wise multiplication
    std::vector<double> res_data(data.size(), 0.0); 
    Tensor out = Tensor(res_data); 

    for (int i = 0; i < res_data.size(); ++i) {
      out.data[i] = this->data[i] * other.data[i]; 
    }

    this->grad = zero_matrix(data.size()); 
    other.grad = zero_matrix(data.size()); 

    Tensor* self_ptr = this; 
    Tensor* other_ptr = &other; 


    out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 
  
    out.backward = [self_ptr, other_ptr] {
      self_ptr->grad = eye_matrix(self_ptr->data.size()); 
      other_ptr->grad = eye_matrix(other_ptr->data.size()); 
      for (int i = 0; i < self_ptr->grad.size(); ++i) {
        self_ptr->grad[i][i] = other_ptr->data[i];
        other_ptr->grad[i][i] = self_ptr->data[i];
      }
    };

    return out; 
  }

  Tensor dot(Tensor &other) {
    // dot product: R^n \times R^n -> R
    
    // compute dot product
    float res = 0.0; 
    for (int i = 0; i < data.size(); ++i) {
      res += data[i] * other.data[i];
    }
    std::vector <double> res_data {res}; 

    // if this/other is leaf node, then initialize 
    // the gradients now we know shape of Jacobian
    this->grad = zero_matrix(1, data.size()); 
    other.grad = zero_matrix(1, data.size()); 

    Tensor out = Tensor(res_data); 
    
    Tensor* self_ptr = this; 
    Tensor* other_ptr = &other; 

    out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 

    out.backward = [self_ptr, other_ptr] {
      for (int i = 0; i < self_ptr->data.size(); ++i) {
        self_ptr->grad[0][i] = other_ptr->data[i];
        other_ptr->grad[0][i] = self_ptr->data[i];
      }
    };

    return out; 
  }

};
