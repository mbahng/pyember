#include <cassert> 
#include <vector>

class BaseTensor {

  // Abstract class for all Tensor objects. 
  public: 
    std::vector<double> storage_; 
    std::vector<size_t> shape_; 
   
    // return references since we don't need to copy
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<double>& data() const { return storage_; } 

    // No constructor since this is an abstract class 
};

class GradTensor : public BaseTensor {
  public: 
    GradTensor(std::vector<double> data, std::vector<size_t> shape) {
      this->storage_ = data; 
      this->shape_ = shape;  
    }

};

class Tensor : public BaseTensor { 
  public: 
    Tensor(std::vector<double> data, std::vector<size_t> shape) {
      this->storage_ = data; 
      this->shape_ = shape;  
    }

};
