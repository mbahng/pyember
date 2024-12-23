#include <vector>
#include <cassert> 
#include "utils.h"
#include "Tensor.h"

int shape_to_length(std::vector<size_t> shape) { 
  int len = 1; 
  for (int i = 0; i < shape.size(); ++i) {
    len *= shape[i]; 
  }
  return len; 
}

size_t prod(std::vector<size_t> input) {
  int product = 1; 
  for (const auto& num : input) {
    product *= num; 
  }
  return product; 
}

void array_matches_shape(
  std::vector<double> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 1); 
  assert(data.size() == shape[0]);
}

void array_matches_shape(
  std::vector<std::vector<double>> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 2); 
  assert(data.size() == shape[0]); 

  for (int i = 0; i < shape[0]; i++) {
    assert(data[i].size() == shape[1]); 
  }
}

void array_matches_shape(
  std::vector<std::vector<std::vector<double>>> data, 
  std::vector<size_t> shape
  ) { 
  assert(shape.size() == 3); 
  assert(data.size() == shape[0]); 

  for (int i = 0; i < shape[0]; i++) {
    assert(data[i].size() == shape[1]); 
    for (int j = 0; j < shape[1]; j++) {
      assert(data[i][j].size() == shape[2]);
    }
  }
}

// Helper function to increment the indices
bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    indices[i]++;
    if (indices[i] < shape[i]) {
      return true;  // Successfully incremented
    }
    indices[i] = 0;  // Reset this position and continue with next position
  }
  return false;  // We've gone through all possibilities
}

// Function to generate all possible vectors
std::vector<std::vector<size_t>> generate_all_indices(const std::vector<size_t>& shape) {
  std::vector<std::vector<size_t>> result;
  
  // Calculate total number of combinations
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }
  result.reserve(total);  // Reserve space for efficiency
  
  // Start with all zeros
  std::vector<size_t> current(shape.size(), 0);
  
  // Add first combination
  result.push_back(current);
  
  // Generate all other combinations
  while (increment_indices(current, shape)) {
    result.push_back(current);
  }
 
  if (result.size() > 0) {
    return result;
  } 
  else {
    return {{}};
  }
}

std::vector<std::vector<size_t>> split_indices(const std::vector<size_t> shape, size_t idx) {
  std::vector<std::vector<size_t>> result(2);
  
  // First part: [0, idx)
  result[0].assign(shape.begin(), shape.begin() + idx);
  
  // Second part: [idx, end)
  result[1].assign(shape.begin() + idx, shape.end());
  return result; 
}

std::vector<double> range(int l, int u, int s) {
  std::vector<double> res; 
  for (int p = l; p < u; p += s) {
    res.push_back(p);
  }
  return res; 
}

std::vector<double> range(int u, int s) {
  return range(0, u, s);
} 

namespace Integrity {

  Shape compat(Tensor* t1, Tensor* t2) {  
    std::vector<size_t> s1 = t1->shape(); 
    std::vector<size_t> s2 = t2->shape(); 
    if (s1.size() == s2.size()) { 
      if (s1 != s2) {
        throw std::logic_error("Shapes do not match. ");
      }
      return Shape{s1, {}, s1, 0};
    }
    else if (s1.size() > s2.size()) {
      if (std::vector<size_t>(s1.end() - s2.size(), s1.end()) != s2) {
        throw std::logic_error("Shapes do not match. ");
      } 
      return Shape{
        s1, 
        std::vector<size_t>(s1.begin(), s1.end() - s2.size()), 
        std::vector<size_t>(s2), 
        s1.size() - s2.size()
      };
    }
    else {
      if (std::vector<size_t>(s2.end() - s1.size(), s2.end()) != s1) {
        throw std::logic_error("Shapes do not match. ");
      }
      return Shape{
        s2, 
        std::vector<size_t>(s2.begin(), s2.end() - s1.size()), 
        std::vector<size_t>(s1), 
        s2.size() - s1.size()
      };
    }
  } 

  Shape compat(GradTensor* t1, GradTensor* t2) {
    std::vector<size_t> s1 = t1->shape(); 
    std::vector<size_t> s2 = t2->shape();  
    size_t pidx1 = t1->pidx(); 
    size_t pidx2 = t2->pidx(); 
    if (s1.size() == s2.size()) { 
      if (s1 != s2) {
        throw std::logic_error("Shapes do not match. ");
      } 
      else if (pidx1 != pidx2) {
        throw std::logic_error("Pivots do not match. ");
      }
      return Shape{s1, {}, s1, 0, pidx1};
    } 
    else if (s1.size() > s2.size()) {
      if (std::vector<size_t>(s1.end() - s2.size(), s1.end()) != s2) {
        throw std::logic_error("Shapes do not match. ");
      }
      else if (s1.size() - s2.size() != pidx1 - pidx2) {
        throw std::logic_error("Pivots do not match. ");
      }
      return Shape{
        s1, 
        std::vector<size_t>(s1.begin(), s1.end() - s2.size()), 
        std::vector<size_t>(s2), 
        s1.size() - s2.size(), 
        pidx1
      };
    }
    else {
      if (std::vector<size_t>(s2.end() - s1.size(), s2.end()) != s1) {
        throw std::logic_error("Shapes do not match. ");
      }
      else if (s2.size() - s1.size() != pidx2 - pidx1) {
        throw std::logic_error("Pivots do not match. ");
      }
      return Shape{
        s2, 
        std::vector<size_t>(s2.begin(), s2.end() - s1.size()), 
        std::vector<size_t>(s1), 
        s2.size() - s1.size(), 
        pidx2
      };
    }
  }

  Shape compat(GradTensor* t1, Tensor* t2) {
    std::vector<size_t> s1 = t1->shape(); 
    std::vector<size_t> s2 = t2->shape();  
    if (s1.size() == s2.size()) { 
      if (s1 != s2) {
        throw std::logic_error("Shapes do not match. ");
      } 
      return Shape{s1, {}, s1, 0};
    } 
    else if (s1.size() > s2.size()) {
      if (std::vector<size_t>(s1.end() - s2.size(), s1.end()) != s2) {
        throw std::logic_error("Shapes do not match. ");
      }
      return Shape{
        s1, 
        std::vector<size_t>(s1.begin(), s1.end() - s2.size()), 
        std::vector<size_t>(s2), 
        s1.size() - s2.size(), 
      };
    }
    else {
      if (std::vector<size_t>(s2.end() - s1.size(), s2.end()) != s1) {
        throw std::logic_error("Shapes do not match. ");
      }
      return Shape{
        s2, 
        std::vector<size_t>(s2.begin(), s2.end() - s1.size()), 
        std::vector<size_t>(s1), 
        s2.size() - s1.size(), 
      };
    }
  } 

  Shape compat(Tensor* t1, GradTensor* t2) {
    return compat(t2, t1); 
  }

  Shape matmul_compat(GradTensor* t1, GradTensor* t2) {  

    if (t1->bidx_ > 0 && t2->bidx_ > 0) { 
      if (t1->bshape() != t2->bshape()) {
        throw std::logic_error("You are multiplying two batches. This is not allowed."); 
      }
    } 

    std::vector<size_t> b1 = t1->bidx_ > 0 ? t1->bshape() : t2->bshape(); 

    std::vector<size_t> s1 = t1->shape(); 
    std::vector<size_t> s2 = t2->shape(); 
    std::vector<size_t> L1 = std::vector<size_t> (s1.begin() + t1->bidx(), s1.begin() + t1->pidx());
    std::vector<size_t> R1 = std::vector<size_t> (s1.begin() + t1->pidx(), s1.end()); 
    std::vector<size_t> L2 = std::vector<size_t> (s2.begin() + t2->bidx(), s2.begin() + t2->pidx()); 
    std::vector<size_t> R2 = std::vector<size_t> (s2.begin() + t2->pidx(), s2.end()); 

    if (R1 != L2) {
      std::ostringstream msg;
      msg << "Dimensions to be contracted are not equal: left (";  
      for (size_t s : R1) { msg << " " << s; }
      msg << " ), right ( ";
      for (size_t s : L2) { msg << " " << s; }
      msg << " )";
      throw std::logic_error(msg.str());
    }

    std::vector<size_t> newshape = concat(b1, L1, R2); 
    return Shape{newshape, b1, {}, b1.size(), b1.size() + L1.size()};
  }

  Shape matmul_compat(Tensor* t1, Tensor* t2) {
    std::vector<size_t> s1 = t1->shape(); 
    std::vector<size_t> s2 = t2->shape(); 

    if (s1.size() >= s2.size()) {
      if (s2.size() != 2) {
        std::string this_shape = ""; 
        for (auto s : t1->shape()) { this_shape += " " + std::to_string(s); } 
        std::string other_shape = ""; 
        for (auto s : t2->shape()) { other_shape += " " + std::to_string(s); } 
        throw std::logic_error("Only rank-2 tensors x rank-2 tensors are supported. \n"
            "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
      } 

      if (s1[s1.size()-1] != s2[0]) {
        std::string this_shape = ""; 
        for (auto s : t1->shape()) { this_shape += " " + std::to_string(s); } 
        std::string other_shape = ""; 
        for (auto s : t2->shape()) { other_shape += " " + std::to_string(s); } 
        throw std::logic_error("The dimension of the contracted rank does not match. \n"
            "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
      } 

      std::vector<size_t> new_shape = concat(
        std::vector<size_t>(s1.begin(), s1.end()-1), 
        std::vector<size_t>(s2.end()-1, s2.end())
      );

      return Shape{
        new_shape, 
        std::vector<size_t>(s1.begin(), s1.end()-2), 
        std::vector<size_t>(new_shape.end()-2, new_shape.end()), 
        s1.size() - 2, 
        s1.size() - 1
      };
    } 
    else {
      if (s1.size() != 2) {
        std::string this_shape = ""; 
        for (auto s : t1->shape()) { this_shape += " " + std::to_string(s); } 
        std::string other_shape = ""; 
        for (auto s : t2->shape()) { other_shape += " " + std::to_string(s); } 
        throw std::logic_error("Only rank-2 tensors x rank-2 tensors are supported. \n"
            "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
      } 

      if (s1[1] != s2[s2.size()-2]) {
        std::string this_shape = ""; 
        for (auto s : t1->shape()) { this_shape += " " + std::to_string(s); } 
        std::string other_shape = ""; 
        for (auto s : t2->shape()) { other_shape += " " + std::to_string(s); } 
        throw std::logic_error("The dimension of the contracted rank does not match. \n"
            "Attempting to multiply (" + this_shape + " ) and (" + other_shape + " )");
      }

      std::vector<size_t> new_shape = concat(
        std::vector<size_t>(s2.begin(), s2.end() - 2), 
        std::vector<size_t>{s1[0]}, 
         std::vector<size_t>{s2[s2.size() - 1]}
      );

      return Shape{
        new_shape, 
        std::vector<size_t>(s2.begin(), s2.end() - 2),  
        std::vector<size_t>(new_shape.end()-2, new_shape.end()), 
        s2.size() - 2, 
        s2.size() - 1
      };
    }
    
  }

};

