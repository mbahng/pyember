#include "common.hpp"

void init_scalartensor_binding(py::module_ &m) {
  py::class_<ScalarTensor, Tensor>(m, "ScalarTensor") 
  .def(py::init<>())
  .def(py::init<double>(), py::arg("data"))
  .def(py::init<std::vector<double>>(), py::arg("data"))
  .def("copy", 
      [](ScalarTensor &a) {
          return a.copy();
      }
    )
  .def("item", 
      [](ScalarTensor &a) {
          return a.item();
      }
    )
;}

