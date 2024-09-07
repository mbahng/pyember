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
#include <eigen3/Eigen/Dense>

int shape_to_length(std::vector<int> shape); 

class Tensor {
public: 
  std::vector<double> data; 
  std::vector<int> shape; 
  int length; 

  // Constructors
  // Should be passed by value? Since we want it to be independent. 
  
  // initializes an array of size (...)
  Tensor(std::vector<int> input_shape) : shape(input_shape) {
    int len = shape_to_length(input_shape); 
    std::vector<double> res(len, 0.); 
    data = res; 
    length = len; 
  }

  // automatic reshape 1D arrays into (...) arrays
  Tensor(std::vector<double> input_data, std::vector<int> input_shape) {
    int len = shape_to_length(input_shape); 
    if (len != input_data.size()) {
      std::cout << len; 
      std::cout << input_data.size(); 
      throw std::logic_error("The shape does not match the size of the array."); 
    }
    data = input_data; 
    shape = input_shape; 
    length = len; 
  }

  // constructor for 1D arrays
  Tensor(std::vector<double> input_data) {
    length = input_data.size(); 
    data = input_data; 
    shape = std::vector<int> {length};
  }

  // constructor for 2D arrays
  Tensor(std::vector<std::vector<double>> input_data) {
    // check first if viable size 
    shape = std::vector<int> {(int)input_data.size(), (int)input_data[0].size()}; 
    for (int i = 0; i < shape[0]; ++i) {
      if (input_data[i].size() != shape[1]) {
        throw std::logic_error("Not a viable size since all arrays are not the same length."); 
      }
    }
    length = shape_to_length(shape); 
    std::vector<double> res(length); 
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        res[shape[1] * i + j] = input_data[i][j]; 
      }
    }
    data = res; 
  }

  // constructor for 3D arrays
  Tensor(std::vector<std::vector<std::vector<double>>> input_data) {
    shape = std::vector<int> {
      (int)input_data.size(), 
      (int)input_data[0].size(), 
      (int)input_data[0][0].size()
    }; 
    // check first if viable size
    for (int i = 0; i < shape[0]; ++i) {
      if (input_data[i].size() != shape[1]) {
        throw std::logic_error("Not a viable size since all arrays are not the same length."); 
      }
      for (int j = 0; j < shape[1]; ++j) {
        if (input_data[i][j].size() != shape[2]) {
          throw std::logic_error("Not a viable size since all arrays are not the same length."); 
        }
      }
    }
    length = shape_to_length(shape); 
    std::vector<double> res(length); 
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        for (int k = 0; k < shape[2]; ++k) {
          res[(shape[1] * shape[2] * i) + (shape[2] * j) + k] = input_data[i][j][k]; 
        }
      }
    }
    data = res; 
  }

  int dimension() {
    return length; 
  }

  // Equality operator
  bool operator==(const Tensor& other) const {
    // First, compare shapes
    if (this->shape != other.shape) {
        return false;
    }

    // Then, compare data
    if (this->data.size() != other.data.size()) {
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

  operator std::string() const { 
    /* std::cout << std::fixed << std::setprecision(2);  */
    
    std::ostringstream oss; 

    if (shape.size() == 1) {
      for (int i = 0; i < length; ++i) {
        oss << std::setw(4) << data[i]; 
      }
    }
    else if (shape.size() == 2) {
      int rows = shape[0]; 
      int cols = shape[1]; 
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          oss << std::setw(4) << data[(i * cols) + j]; 
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
              oss << std::setw(4) << data[start + (i * cols) + j]; 
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

    Tensor result(new_shape);

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

  // Tensor Operations 
  Tensor add(Tensor b) {
    if (b.shape == std::vector<int>{1, 1}) {
      return this->add(b.data[0]); 
    }
    else if (shape == b.shape) {
      std::vector<double> sum(length); 
      for (int i = 0; i < length; ++i) {
        sum[i] = data[i] + b.data[i]; 
      }
      return Tensor(sum, shape); 
    }
    else if (shape.size() == b.shape.size() + 1) {
      for (int i = 0; i < b.shape.size(); ++i) {
        if (shape[i+1] != b.shape[i]) {
          throw std::logic_error("Shapes do not match."); 
        }
      }
      std::vector<double> sum(length); 
      for (int i = 0; i < length; ++i) {
        sum[i] = data[i] + b.data[i / b.length]; 
      }
      return Tensor(sum, shape); 
    }
    else if (shape.size() + 1 == b.shape.size()) {
      return b.add(*this); 
    }
    else {
      throw std::logic_error("Shapes do not match."); 
    }
  }

  Tensor add(double b) {
    std::vector<double> sum(length); 
    for (int i = 0; i < length; ++i) {
      sum[i] = data[i] + b; 
    }
    return Tensor(sum, shape); 
  }

  Tensor sub(Tensor b) {
    if (b.shape == std::vector<int>{1, 1}) {
      return this->sub(b.data[0]); 
    }
    else if (shape == b.shape) {
      std::vector<double> sum(length); 
      for (int i = 0; i < length; ++i) {
        sum[i] = data[i] - b.data[i]; 
      }
      return Tensor(sum, shape); 
    }
    else if (shape.size() == b.shape.size() + 1) {
      for (int i = 0; i < b.shape.size(); ++i) {
        if (shape[i+1] != b.shape[i]) {
          throw std::logic_error("Shapes do not match."); 
        }
      }
      std::vector<double> sum(length); 
      for (int i = 0; i < length; ++i) {
        sum[i] = data[i] - b.data[i / b.length]; 
      }
      return Tensor(sum, shape); 
    }
    else if (shape.size() + 1 == b.shape.size()) {
      return b.sub(*this); 
    }
    else {
      throw std::logic_error("Shapes do not match."); 
    }
  }

  Tensor sub(double b) {
    std::vector<double> sum(length); 
    for (int i = 0; i < length; ++i) {
      sum[i] = data[i] - b; 
    }
    return Tensor(sum, shape); 
  }

  Tensor mul(double b) {
    std::vector<double> res(length); 
    for (int i = 0; i < length; ++i) {
      res[i] = data[i] * b; 
    }
    return Tensor(res, shape); 
  }

  Tensor mul(Tensor b) {
    if (shape == b.shape) {
      std::vector<double> res(length); 
      for (int i = 0; i < length; ++i) {
        res[i] = data[i] * b.data[i]; 
      }
      return Tensor(res, shape); 
    }
    else {
      throw std::logic_error("Shapes do not match for element-wise multiplication."); 
    }
  }

  Tensor pow(double b) {
    std::vector<double> powers(length); 
    for (int i = 0; i < length; ++i) {
      powers[i] = std::pow(data[i], b) ;
    }
    return Tensor(powers, shape); 
  }

  float sum() {
    float res = 0.0; 
    for (int i = 0; i < length; ++i) {
      res += data[i]; 
    }
    return res; 
  }

  float mean() {
    return this->sum() / this->length; 
  }

  float norm() {
    float res = 0.0; 
    for (int i = 0; i < length; ++i) {
      res += data[i] * data[i]; 
    }
    return std::sqrt(res); 
  }

  float dot(Tensor other) {
    if (other.shape != shape) {
      throw std::logic_error("Shapes do not match."); 
    }
    float res = 0.0; 
    for (int i = 0; i < length; ++i) {
      res += data[i] * other.data[i]; 
    }
    return res; 
  } 

  Tensor transpose(int dim1 = 0, int dim2 = 1) const {
    if (dim1 < 0 || dim2 < 0 || dim1 >= shape.size() || dim2 >= shape.size()) {
      throw std::out_of_range("Dimension indices out of range");
    }

    if (dim1 == dim2) {
        return *this;  // No change needed if dimensions are the same
    }

    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);

    Tensor result(new_shape);

    // Calculate strides for the original tensor
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Helper function to convert flat index to multi-dimensional indices
    auto flat_to_indices = [&](int flat_index, const std::vector<int>& shape) {
        std::vector<int> indices(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            indices[i] = flat_index / strides[i];
            flat_index %= strides[i];
        }
        return indices;
    };

    // Helper function to convert multi-dimensional indices to flat index
    auto indices_to_flat = [&](const std::vector<int>& indices, const std::vector<int>& shape) {
        int flat_index = 0;
        for (int i = 0; i < shape.size(); ++i) {
            flat_index += indices[i] * strides[i];
        }
        return flat_index;
    };

    // Perform the transpose
    for (int i = 0; i < length; ++i) {
        auto indices = flat_to_indices(i, shape);
        std::swap(indices[dim1], indices[dim2]);
        int new_index = indices_to_flat(indices, new_shape);
        result.data[new_index] = data[i];
    }

    return result;
    }

  Tensor reshape(const std::vector<int> new_shape) const {
    // Calculate the total number of elements in the new shape
    int new_total_elements = 1;
    for (int dim : new_shape) {
      if (dim < 0) {
        throw std::invalid_argument("New shape dimensions must be non-negative");
      }
      new_total_elements *= dim;
    }

    // Check if the new shape is compatible with the current data
    if (new_total_elements != this->length) {
      throw std::invalid_argument("New shape must have the same total number of elements as the current shape");
    }

    return Tensor(data, new_shape); 
  }

  Tensor matmul(const Tensor& other) const {
    // Check if the tensors are at least 2D
    if (this->shape.size() < 2 || other.shape.size() < 2) {
      throw std::invalid_argument("Both tensors must be at least 2D for matrix multiplication");
    }

    // Check if the last dimension of this tensor matches the second-to-last dimension of other
    if (this->shape.back() != other.shape[other.shape.size() - 2]) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    // Determine the batch size (if any)
    int batch_size = 1;
    for (size_t i = 0; i < this->shape.size() - 2; ++i) {
        if (i < other.shape.size() - 2) {
            if (this->shape[i] != other.shape[i]) {
                throw std::invalid_argument("Incompatible batch dimensions");
            }
            batch_size *= this->shape[i];
        } else {
            batch_size *= this->shape[i];
        }
    }

    // Determine the dimensions of the result
    std::vector<int> result_shape;
    for (size_t i = 0; i < std::max(this->shape.size(), other.shape.size()) - 2; ++i) {
        result_shape.push_back(i < this->shape.size() - 2 ? this->shape[i] : other.shape[i]);
    }
    result_shape.push_back(this->shape[this->shape.size() - 2]);
    result_shape.push_back(other.shape.back());

    Tensor result(result_shape);

    // Perform batch matrix multiplication
    int m = this->shape[this->shape.size() - 2];
    int n = this->shape[this->shape.size() - 1];
    int p = other.shape.back();

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += this->data[batch * m * n + i * n + k] * other.data[batch * n * p + k * p + j];
                }
                result.data[batch * m * p + i * p + j] = sum;
            }
        }
    }

    return result;
    }
}; 

Tensor gaussian(std::vector<int> shape, double mean = 0.0, double stddev = 1.0); 

Tensor uniform(std::vector<int> shape, double min = 0.0, double max= 1.0); 
