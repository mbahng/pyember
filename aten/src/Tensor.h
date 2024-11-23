#include <cassert>
#include <algorithm> 
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath> 
#include <set>


int shape_to_length(std::vector<size_t> shape);

class Tensor; 
class GradTensor; 

class BaseTensor {
  // Abstract class for all Tensor objects. 
  // Constructor should not be made here 
  public: 
    std::vector<double> storage_; 
    std::vector<size_t> shape_; 

    struct Slice {
        size_t start;
        size_t stop;
        size_t step;
        
        Slice(size_t start_ = 0, 
              size_t stop_ = std::numeric_limits<size_t>::max(), 
              size_t step_ = 1)
            : start(start_), stop(stop_), step(step_) {}
    };

    virtual std::string type() const { return "BaseTensor"; } 
    virtual std::string dtype() const { return "double"; }
    virtual ~BaseTensor() = default;
  
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<double>& data() const { return storage_; } 
    BaseTensor& reshape(std::vector<size_t> new_shape);

    virtual bool operator==(BaseTensor& other) const; 
    virtual bool operator!=(BaseTensor& other) const;  
    operator std::string() const; 

    // Tensor Math Operations cannot be virtualized 
    
    // Index access
    virtual double at(const std::vector<size_t>& indices) const;
    virtual double& at(const std::vector<size_t>& indices);

    // Slicing
    virtual std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const;

  protected:
    // Helper methods
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    void validate_indices(const std::vector<size_t>& indices) const;
    std::vector<size_t> calculate_slice_shape(const std::vector<Slice>& slices) const;
    void copy_slice_data(const std::vector<Slice>& slices,
                        std::vector<size_t>& current_indices,
                        size_t current_dim,
                        std::vector<double>& result_storage) const;
};

class GradTensor : public BaseTensor {
  public: 
    size_t pivot_; 

    // Constrcutors
    GradTensor(); 
    GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t pivot); 
    GradTensor(std::vector<size_t> shape, size_t pivot); 
    std::string type() const override { return "GradTensor"; }
    size_t pivot() const { return pivot_; } 
    
    static GradTensor eye(size_t n, size_t pivot = 1); 

    // utils 
    bool operator==(GradTensor& other) const; 
    bool operator!=(GradTensor& other) const;  
    GradTensor copy() const; 

    GradTensor add(GradTensor& other); 
    Tensor add(Tensor& other); 
    GradTensor sub(GradTensor& other); 
    Tensor sub(Tensor& other); 
    GradTensor mul(GradTensor& other); 
    Tensor mul(Tensor& other); 
    GradTensor matmul(GradTensor& other); 

    double at(const std::vector<size_t>& indices) const override {
        return BaseTensor::at(indices);
    }

    double& at(const std::vector<size_t>& indices) override {
        return BaseTensor::at(indices);
    }

    std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const override {
        auto base_result = BaseTensor::slice(slices);
        return std::make_unique<GradTensor>(base_result->storage_, base_result->shape_, pivot_);
    }
};

class Tensor : public BaseTensor { 
  public: 
    GradTensor grad = GradTensor();                     // the Jacobian rather than gradient
    std::vector<Tensor*> prev = std::vector<Tensor*>(); // previous nodes used to compute this tensor, if any 
    std::function<void()> backward;                     // function for filling in gradients of this tensor

    // constructors
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

    std::string type() const override { return "Tensor"; }
    virtual Tensor& reshape(std::vector<size_t> new_shape);
    Tensor copy() const; 

    // backprop functions 
    void build_topo(Tensor* v, std::set<Tensor*>& visited, std::vector<Tensor*>& topo); 
    std::vector<Tensor*> backprop(bool intermediate); 

    Tensor add(Tensor& other); 
    Tensor add(GradTensor& other); 
    Tensor sub(Tensor& other); 
    Tensor sub(GradTensor& other); 
    Tensor mul(Tensor& other); 
    Tensor mul(GradTensor& other); 
    Tensor matmul(Tensor& other); 

    Tensor sum(); 

    double at(const std::vector<size_t>& indices) const override {
        return BaseTensor::at(indices);
    }

    double& at(const std::vector<size_t>& indices) override {
        return BaseTensor::at(indices);
    }

    std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const override {
        auto base_result = BaseTensor::slice(slices);
        return std::make_unique<Tensor>(base_result->storage_, base_result->shape_);
    }
};
