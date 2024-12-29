#include "../common.hpp"
#include "../../src/Tensor/Tensor.h"

void init_gradtensor_binding(py::module_ &m) {
  py::class_<GradTensor, BaseTensor>(m, "GradTensor")

    // base
    .def_property_readonly("pidx", &GradTensor::pidx)
    .def_property_readonly("type", &BaseTensor::type)
    .def_property_readonly("dtype", &BaseTensor::dtype)

    // constructor
    .def(py::init([]() {
        return new GradTensor();
      }))
    .def(py::init([](std::vector<double> data, std::vector<size_t> shape, size_t bidx, size_t pidx) {
        return new GradTensor(data, shape, bidx, pidx);
      }))
    .def(py::init([](std::vector<size_t> shape, size_t bidx, size_t pidx) {
        return new GradTensor(shape, bidx, pidx);
      }))
    .def_static("eye", &GradTensor::eye, 
      py::arg("n"), py::arg("bidx"), py::arg("pidx"))

    // string 
    .def("__repr__", &GradTensor::operator std::string, py::is_operator())
    .def("__str__", &GradTensor::operator std::string, py::is_operator())
    .def("meta", &GradTensor::meta)
    
    // comparison
    .def("__eq__", &GradTensor::operator==, py::is_operator())
    .def("__ne__", &GradTensor::operator!=, py::is_operator()) 
    .def("__lt__", &GradTensor::operator<, py::is_operator()) 
    .def("__gt__", &GradTensor::operator>, py::is_operator()) 
    .def("__le__", &GradTensor::operator<=, py::is_operator()) 
    .def("__ge__", &GradTensor::operator>=, py::is_operator()) 

    // index 
    
    // shape 
    .def("copy", &GradTensor::copy)
    .def("reshape", 
        static_cast<GradTensor* (GradTensor::*)(std::vector<size_t>, bool)>(&GradTensor::reshape),
        py::arg("shape"), 
        py::arg("inplace") = false 
      )
    .def("transpose", 
        static_cast<GradTensor* (GradTensor::*)(const std::vector<size_t> &axes)>(&GradTensor::transpose)
      )

    // algebra
    .def("__neg__", [](GradTensor *a) { double c = -1.; return a->mul(c); })
    .def("__add__", [](GradTensor *a, Tensor *b) { return a->add(b); })
    .def("__radd__", [](GradTensor *a, Tensor *b) { return b->add(a); })
    .def("__add__", [](GradTensor *a, GradTensor *b) { return a->add(b); })
    .def("__radd__", [](GradTensor *a, GradTensor *b) { return b->add(a); })
    .def("__add__", [](GradTensor *a, double b) { return a->add(b); })
    .def("__radd__", [](GradTensor *a, double b) { return a->add(b); })
    .def("__sub__", [](GradTensor *a, Tensor *b) { return a->sub(b); })
    .def("__rsub__", [](GradTensor *a, Tensor *b) { return b->sub(a); })
    .def("__sub__", [](GradTensor *a, GradTensor *b) { return a->sub(b); })
    .def("__rsub__", [](GradTensor *a, GradTensor *b) { return b->sub(a); })
    .def("__sub__", [](GradTensor *a, double b) { return a->sub(b); })
    .def("__mul__", [](GradTensor *a, Tensor *b) { return a->mul(b); })
    .def("__rmul__", [](GradTensor *a, Tensor *b) { return b->mul(a); })
    .def("__mul__", [](GradTensor *a, GradTensor *b) { return a->mul(b); })
    .def("__rmul__", [](GradTensor *a, GradTensor *b) { return b->mul(a); })
    .def("__mul__", [](GradTensor *a, double b) { return a->mul(b); })
    .def("__rmul__", [](GradTensor *a, double b) { return a->mul(b); })
    .def("__matmul__", [](GradTensor* a, GradTensor* b) { return a->matmul(b); })

    // math
    .def("batchsum", &GradTensor::batchsum)
;}
