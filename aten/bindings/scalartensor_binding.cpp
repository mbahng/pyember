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
        [](ScalarTensor &a, ScalarTensor &b) {
            return a.add(b);
        }
      )
    .def("__radd__", 
        [](ScalarTensor &a, ScalarTensor &b) {
            return a.add(b);
        }
      )
    .def("__add__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.add(b);
        }
      )
    .def("__radd__", 
        [](ScalarTensor &a, Tensor &b) {
            return b.add(a);
        }
      )
    .def("__add__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.add(b);
        }
      )
    .def("__radd__", 
        [](ScalarTensor &a, GradTensor &b) {
            return b.add(a);
        }
      )
    .def("__add__", 
        [](ScalarTensor &a, double &b) {
            return a.add(b);
        }
      )
    .def("__radd__", 
        [](ScalarTensor &a, double &b) {
            ScalarTensor scalar = ScalarTensor(b);
            return scalar.add(a);
        }
      )

    .def("__sub__", 
        [](ScalarTensor &a, ScalarTensor &b) {
            return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](ScalarTensor &a, ScalarTensor &b) {
            return b.sub(a);
        }
      )
    .def("__sub__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](ScalarTensor &a, Tensor &b) {
            return b.sub(a);
        }
      )
    .def("__sub__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](ScalarTensor &a, GradTensor &b) {
            return b.sub(a);
        }
      )
    .def("__sub__", 
        [](ScalarTensor &a, double &b) {
            return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](ScalarTensor &a, double &b) {
            ScalarTensor scalar = ScalarTensor(b);
            return scalar.sub(a);
        }
      )

    .def("__mul__", 
        [](ScalarTensor &a, ScalarTensor &b) {
            return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](ScalarTensor &a, ScalarTensor &b) {
            return b.mul(a);
        }
      )
    .def("__mul__", 
        [](ScalarTensor &a, Tensor &b) {
            return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](ScalarTensor &a, Tensor &b) {
            return b.mul(a);
        }
      )
    .def("__mul__", 
        [](ScalarTensor &a, GradTensor &b) {
            return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](ScalarTensor &a, GradTensor &b) {
            return b.mul(a);
        }
      )
    .def("__mul__", 
        [](ScalarTensor &a, double &b) {
            return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](ScalarTensor &a, double &b) {
            ScalarTensor scalar = ScalarTensor(b); 
            return scalar.mul(a);
        }
      )
;}

