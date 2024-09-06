#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"  // This should be the header file containing your Tensor class

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {

  py::class_<Matrix>(m, "Matrix") 
    .def(py::init([](const std::vector<std::vector<double>>& input) {
      size_t rows = input.size(); 
      size_t cols = input[0].size(); 
      return Matrix(rows, cols); 
    }))
    /* .def(py::init<const Eigen::MatrixXd&>())  */
    /* .def(py::init<const std::vector<std::vector<double>>&>())  */
    .def("__len__", &Matrix::dimension) 
    .def("__getitem__", [](Matrix &t, py::tuple index) {
      if (index.size() != 2)
        throw py::index_error("Invalid number of indices");
      return t.getitem(index[0].cast<size_t>(), index[1].cast<size_t>());
    }); 

  py::class_<Vector>(m, "Vector") 
    .def(py::init<std::vector<double>&>()) 
    .def("__len__", &Vector::dimension) 
    .def("__getitem__", [](Vector &t, size_t index) {
      return t.getitem(index); 
    })
    .def("norm", &Vector::norm)
    .def("dot", &Vector::dot) ; 
} 

