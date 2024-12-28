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
#include <gtest/gtest.h> 

class Tensor; 
class GradTensor; 

struct Slice {
  size_t start;
  size_t stop;
  size_t step;
  
  Slice(size_t start_ = 0, 
    size_t stop_ = std::numeric_limits<size_t>::max(), 
    size_t step_ = 1)
  : start(start_), stop(stop_), step(step_) {}
};

class BaseTensor {
  // Abstract class for all Tensor objects. 
  protected: 
    std::vector<double> _storage; 
    std::vector<size_t> _shape;  
    std::vector<size_t> _bshape; 
    std::vector<size_t> _nbshape; 
    size_t _rank; 
    size_t _bidx;  
    size_t _size; 

    // helper methods
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    void validate_indices(const std::vector<size_t>& indices) const;
    std::vector<size_t> calculate_slice_shape(const std::vector<Slice>& slices) const;
    void copy_slice_data(const std::vector<Slice>& slices,
                        std::vector<size_t>& current_indices,
                        size_t current_dim,
                        std::vector<double>& result_storage) const;

  public:  
    // base.cpp  
    const std::vector<double>& storage() const; 
    const size_t rank() const; 
    const size_t hdim() const; 
    const std::vector<size_t>& shape() const;
    const std::vector<size_t> bshape() const; 
    const std::vector<size_t> nbshape() const; 
    const size_t& bidx() const; 
    const size_t& size() const; 
    virtual std::string type() const;
    virtual std::string dtype() const;
    bool is_scalar() const;
    double item() const; 

    // string.cpp
    virtual operator std::string() const; 
    virtual std::string meta() const; 

    // comparison.cpp
    virtual bool operator==(BaseTensor* other) const; 
    virtual bool operator!=(BaseTensor* other) const;  
 
    // index.cpp
    virtual double at(const std::vector<size_t>& indices) const;
    virtual double& at(const std::vector<size_t>& indices);
    virtual std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const;
};

class GradTensor : public BaseTensor {
  private: 
    size_t _pidx; 

  public: 
    // base.cpp 
    size_t pidx() const; 
    std::string type() const override; 
    std::string dtype() const override; 

    // constructor.cpp
    GradTensor(); 
    GradTensor(double scalar); 
    GradTensor(std::vector<double> storage, std::vector<size_t> shape, size_t bidx, size_t pidx); 
    GradTensor(std::vector<size_t> shape, size_t bidx, size_t pidx); 
    static GradTensor* eye(size_t n, size_t bidx, size_t pidx); 
    // zeros, ones, zeros_like, ones_like, uninitialized (requires not vector but array), random ones

    // string.cpp 
    operator std::string() const override; 
    std::string meta() const override; 

    // comparison.cpp 
    bool operator==(GradTensor* other) const; 
    bool operator!=(GradTensor* other) const;  
    // add < and > comparisons
    
    // index.cpp 
    double at(const std::vector<size_t>& indices) const override;
    double& at(const std::vector<size_t>& indices) override;

    std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const override {
      auto base_result = BaseTensor::slice(slices);
      return std::make_unique<GradTensor>(base_result->storage(), base_result->shape(), this->bidx(), this->pidx());
    }

    // shape.cpp 
    GradTensor* copy() const; 
    GradTensor* reshape(std::vector<size_t> new_shape, bool inplace = false); 
    GradTensor* transpose(const std::vector<size_t>& axes = {});

    // algebra.cpp 
    GradTensor* add(double other); 
    GradTensor* add(GradTensor* other); 
    Tensor* add(Tensor* other); 
    GradTensor* sub(double other); 
    GradTensor* sub(GradTensor* other); 
    Tensor* sub(Tensor* other); 
    GradTensor* mul(double other); 
    GradTensor* mul(GradTensor* other); 
    Tensor* mul(Tensor* other); 
    GradTensor* matmul(GradTensor* other); 

    // math.cpp
    GradTensor* batchsum(); 
};

class Tensor : public BaseTensor { 
  protected: 
    std::vector<Tensor*> _prev = std::vector<Tensor*>();
    std::function<void()> _backward;

  public: 
    // attributes that should be modifiable 
    bool requires_grad; 
    GradTensor* grad = nullptr; 

    // base.cpp 
    std::vector<Tensor*> prev() const; // should expose prev to see previous nodes 
      // but not backward since we don't want to modify gradient calculations 
    std::string type() const override; 
    std::string dtype() const override; 

    // constructor.cpp
    Tensor(double scalar, bool requires_grad = true); 
    Tensor(std::vector<size_t> shape, size_t bidx = 0, bool requires_grad = true);
    Tensor(std::vector<double> storage, std::vector<size_t> shape, size_t bidx = 0, bool requires_grad = true);
    Tensor(std::vector<double> storage, size_t bidx = 0, bool requires_grad = true);
    Tensor(std::vector<std::vector<double>> storage, size_t bidx = 0, bool requires_grad = true);
    Tensor(std::vector<std::vector<std::vector<double>>> storage, size_t bidx = 0, bool requires_grad = true); 
    static Tensor* arange(int start, int stop, int step = 1, bool requires_grad = true);
    static Tensor* linspace(double start, double stop, int numsteps, bool requires_grad = true);
    static Tensor* gaussian(std::vector<size_t> shape, double mean = 0.0, double stddev = 1.0, size_t bidx = 0, bool requires_grad = true);
    static Tensor* uniform(std::vector<size_t> shape, double min = 0.0, double max = 1.0, size_t bidx = 0, bool requires_grad = true);
    static Tensor* ones(std::vector<size_t> shape, size_t bidx = 0, bool requires_grad = true);
    static Tensor* zeros(std::vector<size_t> shape, size_t bidx = 0, bool requires_grad = true); 
    ~Tensor() { _prev.clear(); }

    // string.cpp 
    operator std::string() const override; 
    std::string meta() const override;  

    // comparison.cpp
    bool operator==(Tensor* other) const; 
    bool operator!=(Tensor* other) const;  

    // index.cpp
    double at(const std::vector<size_t>& indices) const override; 
    double& at(const std::vector<size_t>& indices) override;
    std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const override;

    // shape.cpp
    Tensor* copy(bool requires_grad = true) const; 
    Tensor* reshape(std::vector<size_t> new_shape, bool inplace = true, bool requires_grad = true);
    Tensor* squeeze(bool inplace = true, bool requires_grad = true);
    Tensor* squeeze(size_t dim, bool inplace = true, bool requires_grad = true);
    Tensor* unsqueeze(size_t dim, bool inplace = true, bool requires_grad = true);
    Tensor* transpose(const std::vector<size_t>& axes = {}, bool inplace = false, bool requires_grad = true);

    // backprop.cpp
    void build_topo(Tensor* v, std::set<Tensor*>& visited, std::vector<Tensor*>& topo); 
    std::vector<Tensor*> backprop(bool intermediate); 

    // algebra.cpp
    Tensor* add(double other); 
    Tensor* add(Tensor* other); 
    Tensor* add(GradTensor* other); 
    Tensor* iadd(GradTensor* other); 

    Tensor* sub(double other); 
    Tensor* sub(Tensor* other); 
    Tensor* sub(GradTensor* other); 
    Tensor* isub(GradTensor* other); 

    Tensor* mul(double other); 
    Tensor* mul(Tensor* other); 
    Tensor* mul(GradTensor* other); 
    Tensor* imul(GradTensor* other); 

    Tensor* matmul(Tensor* other); 

    // math.cpp
    Tensor* dot(Tensor* other);
    Tensor* sum(); 
    Tensor* sum(size_t dim);
    Tensor* sum(std::vector<size_t> dims); 
    Tensor* relu(); 
    Tensor* pow(double* x); 
};
