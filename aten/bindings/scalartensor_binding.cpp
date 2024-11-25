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
    .def("__add__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.add(b);
        }
      )
    .def("__add__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.add(b);
        }
      )
    .def("__sub__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.sub(b);
        }
      )
    .def("__sub__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.sub(b);
        }
      )
    .def("__mul__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.mul(b);
        }
      )
    .def("__mul__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.mul(b);
        }
      )
;}

