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

std::vector<std::vector<double>> eye_matrix(int n, double k = 1.);

class GradTensor {
  public: 
    std::vector<double> data; 
    std::vector<int> shape; 
    int length; 

  // Constructors 
    GradTensor() {
      this->data = std::vector<double> {{}}; 
      this->shape = {0, 0}; 
      this->length = 0;
    }

    GradTensor(std::vector<double> data, std::vector<int> shape) {
      // assert that shape is consistent with data 
      this->length = shape_to_length(shape); 
      assert(data.size() == this->length); 
      this->shape = shape; 
      this->data = data; 
    }

    GradTensor(std::vector<std::vector<double>> data) {
      this->shape = std::vector<int> {(int)data.size(), (int)data[0].size()}; 
      for (int i = 0; i < shape[0]; ++i) {
        if (data[i].size() != shape[1]) {
          throw std::logic_error("Not a viable size since all arrays are not the same length."); 
        }
      }
      this->length = shape_to_length(shape); 
      std::vector<double> res(length); 
      for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
          res[shape[1] * i + j] = data[i][j]; 
        }
      }
      this->data = res; 
    }

  // Utility Functions
    operator std::string() const { 
      std::ostringstream oss; 
      oss << std::fixed << std::setprecision(2);

      int rows = shape[0]; 
      int cols = shape[1]; 
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          oss << std::setw(8) << data[(i * cols) + j]; 
        }
        oss << '\n'; 
      }
      return oss.str(); 
    }

    GradTensor reshape(std::vector<int> new_shape) {
      // Calculate the total number of elements in the new shape
      int new_length = shape_to_length(new_shape); 
      if (new_length != this->length) {
        throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
      }
      return GradTensor(this->data, new_shape); 
    }

  // Math Operations
    GradTensor matmul(GradTensor& other) {
      // Check if the tensors are at least 2D
      assert(this->shape.size() == 2 || other.shape.size() == 2);

      // Check if the last dimension of this tensor matches the second-to-last dimension of other 
      assert(this->shape[1] == other.shape[0]);

      GradTensor out = GradTensor(std::vector<std::vector<double>> (this->shape[0], std::vector<double>(other.shape[1], 0.))); 
      
      // Perform matrix multiplication
      int m = this->shape[0];
      int n = this->shape[1];
      int p = other.shape[1];

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
          double sum = 0.0;
          for (int k = 0; k < n; ++k) {
            sum += this->data[i * n + k] * other.data[k * p + j];
          }
          out.data[i * p + j] = sum;
        }
      }
      return out;
    }
};

class Tensor {
public: 
  std::vector<double> data; 
  std::vector<int> shape; 
  int length; 

  GradTensor grad = GradTensor();                     // the Jacobian rather than gradient
  std::vector<Tensor*> prev = std::vector<Tensor*>(); // previous nodes used to compute this tensor, if any 
  std::function<void()> backward;                     // function for filling in gradients of this tensor

  // Constructors 
    Tensor(std::vector<double> data, std::vector<int> shape) {
      this->length = shape_to_length(shape); 
      assert(data.size() == this->length); 
      this->shape = shape; 
      this->data = data; 
    }

    Tensor(std::vector<double> data) {
      this->data = data; 
      this->length = data.size();
      this->shape = std::vector<int> { length }; 
    }

    Tensor(std::vector<std::vector<double>> data) {
      // check first if viable size 
      this->shape = std::vector<int> {(int)data.size(), (int)data[0].size()}; 
      for (int i = 0; i < shape[0]; ++i) {
        if (data[i].size() != shape[1]) {
          throw std::logic_error("Not a viable size since all arrays are not the same length."); 
        }
      }
      this->length = shape_to_length(shape); 

      std::vector<double> res(length); 
      for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
          res[shape[1] * i + j] = data[i][j]; 
        }
      }

      this->data = res; 
    }

    Tensor(std::vector<std::vector<std::vector<double>>> data) {
      shape = std::vector<int> {
        (int)data.size(), 
        (int)data[0].size(), 
        (int)data[0][0].size()
      }; 
      // check first if viable size
      for (int i = 0; i < shape[0]; ++i) {
        if (data[i].size() != shape[1]) {
          throw std::logic_error("Not a viable size since all arrays are not the same length."); 
        }
        for (int j = 0; j < shape[1]; ++j) {
          if (data[i][j].size() != shape[2]) {
            throw std::logic_error("Not a viable size since all arrays are not the same length."); 
          }
        }
      }
      // reshape 3D input to 1D array
      length = shape_to_length(shape); 
      std::vector<double> res(length); 
      for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
          for (int k = 0; k < shape[2]; ++k) {
            res[(shape[1] * shape[2] * i) + (shape[2] * j) + k] = data[i][j][k]; 
          }
        }
      }
      this->data = res; 
    }

    static Tensor gaussian(std::vector<int> shape, double mean, double stddev) {
      // Create a unique seed by combining high-resolution time and a counter
      static std::atomic<unsigned long long> seed_counter{0};

      auto now = std::chrono::high_resolution_clock::now();
      auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
      unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32);

      // Create a generator with the unique seed
      std::mt19937 generator(unique_seed);

      // Create a distribution
      std::normal_distribution<double> distribution(mean, stddev);

      // Calculate total number of elements
      int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

      // Create and fill the vector
      std::vector<double> result(length);
      for (int i = 0; i < length; ++i) {
        result[i] = distribution(generator);
      }

      return Tensor(result, shape);
    }

    static Tensor uniform(std::vector<int> shape, double min, double max) {
      // (Use the same unique seeding method as in the gaussian function)
      static std::atomic<unsigned long long> seed_counter{0};

      auto now = std::chrono::high_resolution_clock::now();
      auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
      unsigned long long unique_seed = nanos ^ (seed_counter.fetch_add(1, std::memory_order_relaxed) << 32);

      std::mt19937 generator(unique_seed);
      std::uniform_real_distribution<double> distribution(min, max);

      int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      std::vector<double> result(length);
      for (int i = 0; i < length; ++i) {
        result[i] = distribution(generator);
      }

      return Tensor(result, shape);
    }

    static Tensor ones(std::vector<int> shape) {
      std::vector<double> data(shape_to_length(shape), 1.0); 
      return Tensor(data, shape); 
    }

    static Tensor zeros(std::vector<int> shape) {
      std::vector<double> data(shape_to_length(shape), 0.0); 
      return Tensor(data, shape); 
    }

  // Utility Functions  
    bool operator==(const Tensor& other) const {
      // First, compare shapes
      if (this->shape != other.shape) {
        return false;
      }

      // Compare each element, considering floating-point precision
      const double epsilon = std::numeric_limits<double>::epsilon();
      for (size_t i = 0; i < this->data.size(); ++i) {
        if (std::abs(this->data[i] - other.data[i]) > epsilon) {
          return false;
        }
      }

      return true;
    }

    // Inequality operator (for completeness)
    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }

    Tensor copy() { 
      // gradients are not copied
      return Tensor(this->data, this->shape);
    }

    operator std::string() const { 
      std::ostringstream oss; 
      oss << std::fixed << std::setprecision(2);

      if (shape.size() == 1) {
        for (int i = 0; i < length; ++i) {
          oss << std::setw(8) << data[i]; 
        }
      }
      else if (shape.size() == 2) {
        int rows = shape[0]; 
        int cols = shape[1]; 
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            oss << std::setw(8) << data[(i * cols) + j]; 
          }
          oss << '\n'; 
        }
      }
      else if (shape.size() == 3) {
        int rows = shape[1]; 
        int cols = shape[2]; 

        for (int start = 0; start < length; start += rows * cols) {
          for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                oss << std::setw(8) << data[start + (i * cols) + j]; 
            }
            oss << '\n'; 
          }
          oss << '\n'; 
        }
      }
      else {
        throw std::logic_error("Shape is more than 3 hyperdimensions. This is not supported."); 
      }
      return oss.str(); 
    }

    Tensor reshape(std::vector<int> new_shape) {
      // Calculate the total number of elements in the new shape
      int new_length = shape_to_length(new_shape); 
      if (new_length != this->length) {
        throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
      }

      return Tensor(data, new_shape); 
    }

  // Indexing and slicing 
    const double& at(const std::vector<size_t>& indices) const {
      if (indices.size() != shape.size()) {
        throw std::out_of_range("Number of indices does not match tensor dimensions.");
      }
      size_t flat_index = 0;
      size_t multiplier = 1;
      for (int i = shape.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape[i]) {
          throw std::out_of_range("Index out of bounds.");
        }
        flat_index += indices[i] * multiplier;
        multiplier *= shape[i];
      }
      return data[flat_index];
    }

    double& at(const std::vector<size_t>& indices) {
      return const_cast<double&>(static_cast<const Tensor*>(this)->at(indices));
    }

    class Slice {
      public:
          int start;
          int stop;
          int step;

          Slice(int start_ = 0, int stop_ = -1, int step_ = 1)
              : start(start_), stop(stop_), step(step_) {}
      };

    Tensor slice(const std::vector<Slice>& slices) const {
      std::vector<int> new_shape;
      std::vector<int> start_indices;
      std::vector<int> steps;

      for (size_t i = 0; i < shape.size(); ++i) {
        if (i < slices.size()) {
          const auto& s = slices[i];
          int start = (s.start >= 0) ? s.start : shape[i] + s.start;
          int stop = (s.stop >= 0) ? s.stop : shape[i] + s.stop;
          start = std::max(0, std::min(start, shape[i]));
          stop = std::max(0, std::min(stop, shape[i]));
          int slice_length = (stop - start + s.step - 1) / s.step;
          new_shape.push_back(slice_length);
          start_indices.push_back(start);
          steps.push_back(s.step);
        } 
        else {
          new_shape.push_back(shape[i]);
          start_indices.push_back(0);
          steps.push_back(1);
        }
      }

      Tensor result = Tensor(std::vector<double> (shape_to_length(new_shape), 0));
      result.shape = new_shape;

      std::function<void(size_t, std::vector<size_t>&, std::vector<size_t>&)> copy_data = 
        [&](size_t dim, std::vector<size_t>& src_indices, std::vector<size_t>& dst_indices) {
          if (dim == shape.size()) {
            result.at(dst_indices) = this->at(src_indices);
            return;
          }
          int start = start_indices[dim];
          int step = steps[dim];
          int end = start + new_shape[dim] * step;
          for (int i = start; i < end; i += step) {
            src_indices[dim] = i;
            dst_indices[dim] = (i - start) / step;
            copy_data(dim + 1, src_indices, dst_indices);
          }
        };

      std::vector<size_t> src_indices(shape.size());
      std::vector<size_t> dst_indices(shape.size());
      copy_data(0, src_indices, dst_indices);

      return result;
    }

  // Backpropagation Functions
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
      this->grad = GradTensor(eye_matrix(this->data.size()));
      
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

  // Unary Argument Scalar Math Operations 
    Tensor sum() {
      std::vector<double> out_data {0.0};  
      for (int i = 0; i < this->length; ++i) {
        out_data[0] += this->data[i]; 
      }

      Tensor out = Tensor(out_data); 
      out.prev = std::vector<Tensor*> {this}; 

      this->grad = GradTensor(zero_matrix(1, this->length)); 

      out.backward = [this] {
        for (int i = 0; i < this->length; ++i) {
          (this->grad).data[i] = 1.0;
        }
      };
      return out; 
    }

    Tensor mean() {
      std::vector<double> out_data {0.0};  
      for (int i = 0; i < this->length; ++i) {
        out_data[0] += this->data[i]; 
      }
      out_data[0] /= this->length; 

      Tensor out = Tensor(out_data); 
      out.prev = std::vector<Tensor*> {this}; 

      this->grad = GradTensor(zero_matrix(1, this->length)); 

      out.backward = [this] {
        int t = 1/(this->length);
        for (int i = 0; i < this->length; ++i) {
          (this->grad).data[i] = t;
        }
      };
      return out; 
    }
 
  // Unary Argument Vector Math Operations 
    Tensor pow(double n) { 
      Tensor out = this->copy(); 
      for (int i = 0; i < this->length; ++i) {
        out.data[i] = std::pow(out.data[i], n);
      }

      this->grad = GradTensor(zero_matrix(data.size())); 

      Tensor* self_ptr = this; 

      out.prev = std::vector<Tensor*> {self_ptr}; 
    
      out.backward = [self_ptr, n] {
        self_ptr->grad = GradTensor(eye_matrix(self_ptr->data.size())); 
        for (int i = 0; i < self_ptr->grad.length; ++i) {
          (self_ptr->grad).data[i * self_ptr->data.size() + i] = n * std::pow(self_ptr->data[i], n-1);
        }
      };
      return out; 
    }

  // Binary Argument Scalar Math Operations
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
      this->grad = GradTensor(zero_matrix(1, data.size())); 
      other.grad = GradTensor(zero_matrix(1, data.size())); 

      Tensor out = Tensor(res_data); 
      
      Tensor* self_ptr = this; 
      Tensor* other_ptr = &other; 

      out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 

      out.backward = [self_ptr, other_ptr] {
        for (int i = 0; i < self_ptr->data.size(); ++i) {
          (self_ptr->grad).data[i] = other_ptr->data[i];
          (other_ptr->grad).data[i] = self_ptr->data[i];
        }
      };

      return out; 
    }

  // Binary Argument Vector Math Operations
    Tensor add(Tensor& other) {
      std::vector<double> res_data(this->length, 0.0); 

      for (int i = 0; i < this->length; ++i) {
        res_data[i] = this->data[i] + other.data[i]; 
      }

      Tensor out = Tensor(res_data); 
      out.shape = this->shape; 

      this->grad = GradTensor(zero_matrix(this->length)); 
      other.grad = GradTensor(zero_matrix(other.length)); 

      Tensor* self_ptr = this; 
      Tensor* other_ptr = &other; 

      out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 
    
      out.backward = [self_ptr, other_ptr] {
        for (int i = 0; i < self_ptr->length; ++i) {
          (self_ptr->grad).data[i * self_ptr->length + i] = 1.0;
          (other_ptr->grad).data[i * self_ptr->length + i] = 1.0;
        }
      };
      return out; 
    }

    Tensor add(GradTensor& other) {
      std::vector<double> res_data(this->length, 0.0); 

      for (int i = 0; i < this->length; ++i) {
        res_data[i] = this->data[i] + other.data[i]; 
      }

      Tensor out = Tensor(res_data, this->shape); 
      return Tensor(res_data, this->shape); 
    }

    Tensor sub(Tensor& other) {
      std::vector<double> res_data(data.size(), 0.0); 
      Tensor out = Tensor(res_data); 
      for (int i = 0; i < res_data.size(); ++i) {
        out.data[i] = this->data[i] - other.data[i]; 
      }

      this->grad = GradTensor(zero_matrix(data.size())); 
      other.grad = GradTensor(zero_matrix(data.size())); 

      Tensor* self_ptr = this; 
      Tensor* other_ptr = &other; 

      out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 
    
      out.backward = [self_ptr, other_ptr] {
        self_ptr->grad = GradTensor(eye_matrix(self_ptr->data.size())); 
        other_ptr->grad = GradTensor(eye_matrix(other_ptr->data.size(), -1.0)); 
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

      this->grad = GradTensor(zero_matrix(data.size())); 
      other.grad = GradTensor(zero_matrix(data.size())); 

      Tensor* self_ptr = this; 
      Tensor* other_ptr = &other; 

      out.prev = std::vector<Tensor*> {self_ptr, other_ptr}; 
    
      out.backward = [self_ptr, other_ptr] {
        self_ptr->grad = GradTensor(eye_matrix(self_ptr->data.size())); 
        other_ptr->grad = GradTensor(eye_matrix(other_ptr->data.size())); 
        for (int i = 0; i < self_ptr->grad.length; ++i) {
          (self_ptr->grad).data[i * self_ptr->data.size() + i] = other_ptr->data[i];
          (other_ptr->grad).data[i * self_ptr->data.size() + i] = self_ptr->data[i];
        }
      };

      return out; 
    }

    Tensor matmul(Tensor& other) {
      // Check if the tensors are at least 2D
      assert(this->shape.size() == 2 || other.shape.size() == 2);

      // Check if the last dimension of this tensor matches the second-to-last dimension of other
      assert(this->shape[1] == other.shape[0]);

      // Determine the dimensions of the result
      std::vector<int> result_shape {this->shape[0], other.shape[1]};

      Tensor out(std::vector<double> (shape_to_length(result_shape), 0.0), result_shape);

      // Perform batch matrix multiplication
      int m = this->shape[0];
      int n = this->shape[1];
      int p = other.shape[1];

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
          double sum = 0.0;
          for (int k = 0; k < n; ++k) {
            sum += this->data[i * n + k] * other.data[k * p + j];
          }
          out.data[i * p + j] = sum;
        }
      }

      this->grad = GradTensor(zero_matrix(out.data.size(), this->data.size())); 
      other.grad = GradTensor(zero_matrix(out.data.size(), other.data.size())); 
      
      Tensor* self_ptr = this; 
      Tensor* other_ptr = &other; 

      out.prev = std::vector<Tensor*> {self_ptr, other_ptr};  

      out.backward = [out, self_ptr, other_ptr] {
        for (int z1 = 0; z1 < out.shape[0]; ++z1) {
          for (int z2 = 0; z2 < out.shape[1]; ++z2) {
            // Z[z1, z2] = \sum_* x_{z1, * } y_{ * , z2 }  * from 0 to out.shape[0] 
            int z_idx = z1 * out.shape[1] + z2;
            // update gradients on X first 
            // x goes from (z1, 0), (z1, 1), ..., (z1, out.shape[0])
            for (int i = 0; i < self_ptr->shape[1]; ++i) {
              int x_idx = z1 * out.shape[0] + i;
              int y_idx = i * out.shape[1] + z2; 
              (self_ptr->grad).data[z_idx * self_ptr->data.size() + x_idx] = other_ptr->data[y_idx];
              (other_ptr->grad).data[z_idx * other_ptr->data.size() + y_idx] = self_ptr->data[x_idx];
            }
          }
        }
      };

      return out;
    }

  // Activation Functions
  Tensor relu() {
    std::vector<double> out_data = this->data; 
    for (int i = 0; i < this->length; ++i) {
      if (out_data[i] < 0.) {
        out_data[i] = 0.0;
      }
    }

    Tensor out = Tensor(out_data); 
    out.shape = this->shape; 

    this->grad = GradTensor(zero_matrix(this->length, this->length)); 

    out.prev = std::vector<Tensor*> {this}; 

    out.backward = [this] {
      for (int i = 0; i < this->length; ++i) {
        if ((this->data)[i] >= 0) {
          (this->grad).data[i * this->length + i] = 1.0;
        }
        else {
          (this->grad).data[i * this->length + i] = 0.0; 
        }
      }
    };
    return out; 
  }

};

