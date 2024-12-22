#pragma once
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

class Tensor; 
class GradTensor; 
class ScalarTensor; 

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

    virtual bool operator==(BaseTensor& other) const; 
    virtual bool operator!=(BaseTensor& other) const;  
    virtual operator std::string() const; 

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
    size_t pidx_; 

    // Constrcutors
    GradTensor(); 
    GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t pidx); 
    GradTensor(std::vector<size_t> shape, size_t pidx); 
    std::string type() const override { return "GradTensor"; }
    size_t pidx() const { return pidx_; }  
    
    static GradTensor* eye(size_t n, size_t pidx); 

    // utils 
    bool operator==(GradTensor& other) const; 
    bool operator!=(GradTensor& other) const;  
    GradTensor* copy() const; 
    operator std::string() const override; 

    // default should be false since we don't want original gradient shapes being corrupted
    GradTensor* reshape(std::vector<size_t> new_shape, bool inplace = false); 

    GradTensor* add(GradTensor* other); 
    Tensor* add(Tensor* other); 
    GradTensor* add(ScalarTensor* other);
    GradTensor* add(double* other);

    GradTensor* sub(GradTensor* other); 
    Tensor* sub(Tensor* other); 
    GradTensor* sub(ScalarTensor* other);
    GradTensor* sub(double* other);

    GradTensor* mul(GradTensor* other); 
    Tensor* mul(Tensor* other); 
    GradTensor* mul(ScalarTensor* other);
    GradTensor* mul(double* other);

    GradTensor* matmul(GradTensor* other); 

    double at(const std::vector<size_t>& indices) const override {
        return BaseTensor::at(indices);
    }

    double& at(const std::vector<size_t>& indices) override {
        return BaseTensor::at(indices);
    }

    std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const override {
        auto base_result = BaseTensor::slice(slices);
        return std::make_unique<GradTensor>(base_result->storage_, base_result->shape_, pidx_);
    }
    // Add to GradTensor class:
    GradTensor& transpose(const std::vector<size_t>& axes = {});
};

class Tensor : public BaseTensor { 
  public: 
    bool has_grad = true; 
    GradTensor* grad = nullptr; 
    std::vector<Tensor*> prev = std::vector<Tensor*>();
    std::function<void()> backward;

    // constructors
    Tensor(std::vector<size_t> shape, bool has_grad = true);
    Tensor(std::vector<double> data, std::vector<size_t> shape, bool has_grad = true);
    Tensor(std::vector<double> data, bool has_grad = true);
    Tensor(std::vector<std::vector<double>> data, bool has_grad = true);
    Tensor(std::vector<std::vector<std::vector<double>>> data, bool has_grad = true); 

    // Destructors 
    ~Tensor() { prev.clear(); }

    static Tensor* arange(int start, int stop, int step = 1, bool has_grad = true);
    static Tensor* linspace(double start, double stop, int numsteps, bool has_grad = true);
    static Tensor* gaussian(std::vector<size_t> shape, double mean = 0.0, double stddev = 1.0, bool has_grad = true);
    static Tensor* uniform(std::vector<size_t> shape, double min = 0.0, double max = 1.0, bool has_grad = true);
    static Tensor* ones(std::vector<size_t> shape, bool has_grad = true);
    static Tensor* zeros(std::vector<size_t> shape, bool has_grad = true); 

    std::string type() const override { return "Tensor"; }
    virtual Tensor* reshape(std::vector<size_t> new_shape, bool inplace = true, bool has_grad = true);
    Tensor* copy(bool has_grad = true) const; 
    Tensor* transpose(const std::vector<size_t>& axes = {}, bool inplace = false, bool has_grad = true);
    operator std::string() const override; 

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

    // backprop functions 
    void build_topo(Tensor* v, std::set<Tensor*>& visited, std::vector<Tensor*>& topo); 
    std::vector<Tensor*> backprop(bool intermediate); 

    Tensor* add(Tensor* other); 
    Tensor* add(GradTensor* other); 
    Tensor* add(ScalarTensor* other); 
    Tensor* add(double* other); 

    Tensor* sub(Tensor* other); 
    Tensor* sub(GradTensor* other); 
    Tensor* sub(ScalarTensor* other); 
    Tensor* sub(double* other); 

    Tensor* mul(Tensor* other); 
    Tensor* mul(GradTensor* other); 
    Tensor* mul(ScalarTensor* other); 
    Tensor* mul(double* other); 

    Tensor* matmul(Tensor* other); 

    Tensor* dot(Tensor* other);
    Tensor* sum(); 
    Tensor* relu(); 
    Tensor* pow(double* x); 
};

class ScalarTensor : public Tensor {
  public: 
    // constructors 
    ScalarTensor(); 
    ScalarTensor(double data); 
    ScalarTensor(std::vector<double> data); 

    ~ScalarTensor() { prev.clear(); } 

    std::string type() const override { return "ScalarTensor"; } 

    ScalarTensor* copy() const; 
    double item() const; 

    Tensor* add(Tensor* other); 
    GradTensor* add(GradTensor* other); 
    ScalarTensor* add(ScalarTensor* other); 
    ScalarTensor* add(double* other); 

    Tensor* sub(Tensor* other); 
    GradTensor* sub(GradTensor* other); 
    ScalarTensor* sub(ScalarTensor* other); 
    ScalarTensor* sub(double* other); 

    Tensor* mul(Tensor* other); 
    GradTensor* mul(GradTensor* other); 
    ScalarTensor* mul(ScalarTensor* other); 
    ScalarTensor* mul(double* other); 
};
