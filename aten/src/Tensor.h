#include <cassert> 
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

int shape_to_length(std::vector<size_t> shape);

class BaseTensor {
  // Abstract class for all Tensor objects. 
  // Constructor should not be made here 
  public: 
    std::vector<double> storage_; 
    std::vector<size_t> shape_; 

    // const (member function qualifier). 
    // Function promises not to modify the object it's called on
    virtual std::string type() const { return "BaseTensor"; } 
    virtual std::string dtype() const { return "double"; }
  
    // Tensor_View
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<double>& data() const { return storage_; } 

    // Tensor_IO
    operator std::string() const; 

    // Tensor_Util
    const bool operator==(BaseTensor& other) const; 
    const bool operator!=(BaseTensor& other) const; 

    // Tensor_View.cpp
    BaseTensor& reshape(std::vector<size_t> new_shape);

    // Tensor_Index.cpp 
    const double& at(const std::vector<size_t>& indices) const; 
    double& at(const std::vector<size_t>& indices); 

};

class GradTensor : public BaseTensor {
  // tensor representing gradients
  public: 
    GradTensor(std::vector<double> data, std::vector<size_t> shape) {
      this->storage_ = data; 
      this->shape_ = shape;  
    }
};

class Tensor : public BaseTensor { 
  public: 
    // Constructors 
    Tensor(std::vector<double> data, std::vector<size_t> shape);
    Tensor(std::vector<double> data);
    Tensor(std::vector<std::vector<double>> data);
    Tensor(std::vector<std::vector<std::vector<double>>> data);

    static Tensor arange(int start, int stop, int step = 1);

    static Tensor linspace(double start, double stop, int numsteps);
    static Tensor gaussian(std::vector<size_t> shape, double mean = 0.0, double stddev = 1.0);
    static Tensor uniform(std::vector<size_t> shape, double min = 0.0, double max = 1.0);
    static Tensor ones(std::vector<size_t> shape);
    static Tensor zeros(std::vector<size_t> shape);

    std::string type() const { return "Tensor"; }
    virtual Tensor& reshape(std::vector<size_t> new_shape);
    Tensor copy(); 
    

    Tensor add(Tensor& other); 
    Tensor sub(Tensor& other); 
    Tensor scamul(Tensor& other); 
};

class ScalarTensor : public Tensor {
  public: 
    ScalarTensor(std::vector<double> data) 
    : Tensor(data, std::vector<size_t>()) {}
};
