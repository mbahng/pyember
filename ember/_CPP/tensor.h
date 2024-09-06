#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <eigen3/Eigen/Dense>

class Matrix {
private: 
  Eigen::MatrixXd data; 

public: 

  const int rows() {
    return data.rows(); 
  }

  const int cols() {
    return data.cols(); 
  }

  Matrix(const size_t rows, const size_t cols) {
    std::cout << "First\n"; 
    data = Eigen::MatrixXd::Zero(rows, cols); 
  }; 

  Matrix(Eigen::MatrixXd input) {
    data = input; 
  }; 

  const std::tuple<int, int> shape() {
    return std::tuple<int, int>(rows(), cols()); 
  }

  const int dimension() {
    return rows() * cols(); 
  }

  const double& getitem(size_t row_idx, size_t col_idx) {
    if (row_idx >= rows() || col_idx >= cols()) {
      throw std::out_of_range("Index out of range");
    }
    return data(row_idx, col_idx); 
  }

  void setitem(size_t row_idx, size_t col_idx, double val) {
    if (row_idx >= rows() || col_idx >= cols()) {
      throw std::out_of_range("Index out of range");
    }
    data(row_idx, col_idx) = val;
  }

  operator std::string() const { 
  std::stringstream ss;
    ss << data;
    return ss.str();
  }

  Matrix add(Matrix other) {
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix(data + other.data); 
  }

  Matrix scalar_mul(double other) {
    return Matrix(other * data); 
  }

  Matrix elem_mul(Matrix other) {
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix(data.cwiseProduct(other.data)); 
  }

  Matrix mat_mul(Matrix other) {
    assert(cols() == other.rows());
    return Matrix(data * other.data); 
  }

  Matrix transpose() {
    return Matrix(data.transpose()); 
  }
}; 

class Vector {
private: 
  std::vector<double> data; 
  size_t length; 

public: 
    Vector(const std::vector<double>& input_data) {
    if (input_data.empty()) {
      throw std::invalid_argument("Input data cannot be empty."); 
    }

    data = input_data; 
    length = input_data.size(); 
  }

  std::tuple<int> shape() {
    return std::tuple<int>(length); 
  }

  int dimension() {
    return length; 
  }

  double& getitem(size_t index) {
    if (index >= length) {
      throw std::out_of_range("Index out of range");
    }
    return data[index];
  }

  float norm() {
    float res = 0.0; 
    for (int i = 0; i < length; ++i) {
      res += data[i] * data[i]; 
    }
    return std::sqrt(res); 
  }

  float dot(Vector other) {
    assert(other.length == length); 
    float res = 0.0; 
    for (int i = 0; i < length; ++i) {
      res += data[i] * other.data[i]; 
    }
    return res; 
  } 
}; 
