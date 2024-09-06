#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h" 

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {

  py::class_<Matrix>(m, "Matrix") 
    .def(py::init([](const size_t rows, const size_t cols) {
      return Matrix(rows, cols); 
    }))

    .def(py::init([](Eigen::MatrixXd input) {
      return Matrix(input); 
    }))

    .def(py::init([](const std::vector<std::vector<double>>& input) {
      size_t rows = input.size(); 
      size_t cols = input[0].size(); 
      Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(rows, cols); 
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          matrix(i, j) = input[i][j]; 
        }
      }
      return Matrix(matrix);
    }))

    .def("__len__", &Matrix::dimension) 

    .def("shape", 
      [](Matrix &t) {
        return py::make_tuple(std::get<0>(t.shape()), std::get<1>(t.shape())); 
      })

    .def("__getitem__", 
      [](Matrix &t, py::tuple index) {
        if (index.size() != 2)
          throw py::index_error("Invalid number of indices");
        return t.getitem(index[0].cast<size_t>(), index[1].cast<size_t>()); 
      })

    .def("__setitem__", 
      [](Matrix &t, py::tuple index, double value) {
        if (index.size() != 2)
          throw py::index_error("Invalid number of indices");
        return t.setitem(index[0].cast<size_t>(), index[1].cast<size_t>(), value);
      })

    .def("__repr__",
      [](Matrix &a) {
        return static_cast<std::string>(a);
      })

    .def("__str__",
      [](Matrix &a) {
        return static_cast<std::string>(a);
      })

    .def("__add__", 
      [](Matrix &a , Matrix &b) {
        return a.add(b); 
      })

    .def("__mul__", 
      [](Matrix &a , Matrix &b) {
        return a.mat_mul(b); 
      })

    .def("__mul__", 
      [](Matrix &a , double b) {
         return a.scalar_mul(b); 
      })

    .def("mat_mul", 
      [](Matrix &a , Matrix &b) {
        return a.mat_mul(b); 
      })

    .def("elem_mul", 
      [](Matrix &a , Matrix &b) {
        return a.elem_mul(b); 
      })

    .def("transpose", 
      [](Matrix &a) {
        return a.transpose(); 
      })
  ;


  py::class_<Vector>(m, "Vector") 
    .def(py::init<std::vector<double>&>()) 
    .def("__len__", &Vector::dimension) 
    .def("__getitem__", [](Vector &t, size_t index) {
      return t.getitem(index); 
    })
    .def("norm", &Vector::norm)
    .def("dot", &Vector::dot) ; 
} 

