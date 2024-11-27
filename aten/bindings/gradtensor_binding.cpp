#include "common.hpp"

void init_gradtensor_binding(py::module_ &m) {
  py::class_<GradTensor, BaseTensor>(m, "GradTensor")
    .def_readwrite("pivot", &GradTensor::pivot_)

    .def(py::init<>())
    .def(py::init<std::vector<double>, std::vector<size_t>, size_t>(),
     py::arg("data"), py::arg("shape"), py::arg("pivot")) 
    .def(py::init<std::vector<size_t>, size_t>(),
     py::arg("shape"), py::arg("pivot")) 
    .def_static("eye", &GradTensor::eye, 
      py::arg("n"), py::arg("pivot"))
    .def("copy", 
        [](GradTensor &a) {
          return a.copy();
        }
      )

    .def("__repr__", &GradTensor::operator std::string, py::is_operator())
    .def("__str__", &GradTensor::operator std::string, py::is_operator())
    .def("__eq__", &GradTensor::operator==, py::is_operator())
    .def("__ne__", &GradTensor::operator!=, py::is_operator()) 

    .def("transpose", 
        [](GradTensor &a, const std::vector<size_t> &axes) {
            return a.transpose();
        }
      )

    .def("__neg__", 
        [](GradTensor &a) {
          double c = -1.;
          return a.mul(c);
        }
      )
    .def("__add__", 
        [](GradTensor &a, ScalarTensor &b) {
          return a.add(b);
        }
      )
    .def("__radd__", 
        [](GradTensor &a, ScalarTensor &b) {
          return b.add(a);
        }
      )
    .def("__add__", 
        [](GradTensor &a, Tensor &b) {
          return a.add(b);
        }
      )
    .def("__radd__", 
        [](GradTensor &a, Tensor &b) {
          return b.add(a);
        }
      )
    .def("__add__", 
        [](GradTensor &a, GradTensor &b) {
          return a.add(b);
        }
      )
    .def("__radd__", 
        [](GradTensor &a, GradTensor &b) {
          return b.add(a);
        }
      )
    .def("__add__", 
        [](GradTensor &a, double &b) {
          return a.add(b);
        }
      )
    .def("__radd__", 
        [](GradTensor &a, double &b) {
          return a.add(b);
        }
      )
    .def("__sub__", 
        [](GradTensor &a, ScalarTensor &b) {
          return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](GradTensor &a, ScalarTensor &b) {
          return b.sub(a);
        }
      )
    .def("__sub__", 
        [](GradTensor &a, Tensor &b) {
          return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](GradTensor &a, Tensor &b) {
          return b.sub(a);
        }
      )
    .def("__sub__", 
        [](GradTensor &a, GradTensor &b) {
          return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](GradTensor &a, GradTensor &b) {
          return b.sub(a);
        }
      )
    .def("__sub__", 
        [](GradTensor &a, double &b) {
          return a.sub(b);
        }
      )
    .def("__rsub__", 
        [](GradTensor &a, double &b) {
          ScalarTensor scalar = ScalarTensor(b); // should prob fix this
          return scalar.sub(a);
        }
      )
    .def("__mul__", 
        [](GradTensor &a, ScalarTensor &b) {
          return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](GradTensor &a, ScalarTensor &b) {
          return b.mul(a);
        }
      )
    .def("__mul__", 
        [](GradTensor &a, Tensor &b) {
          return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](GradTensor &a, Tensor &b) {
          return b.mul(a);
        }
      )
    .def("__mul__", 
        [](GradTensor &a, GradTensor &b) {
          return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](GradTensor &a, GradTensor &b) {
          return b.mul(a);
        }
      )
    .def("__mul__", 
        [](GradTensor &a, double &b) {
          return a.mul(b);
        }
      )
    .def("__rmul__", 
        [](GradTensor &a, double &b) {
          return a.mul(b);
        }
      )
    .def("__matmul__", 
        [](GradTensor* a, GradTensor* b) {
          return a->matmul(b);
        }
      )
;}
