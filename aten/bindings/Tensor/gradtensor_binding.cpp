#include "../common.hpp"
#include "../../src/Tensor/Tensor.h"

void init_gradtensor_binding(py::module_ &m) {
  py::class_<GradTensor, BaseTensor>(m, "GradTensor")

    // base
    .def_property_readonly("pidx", &GradTensor::pidx)
    .def_property_readonly("type", &BaseTensor::type)
    .def_property_readonly("dtype", &BaseTensor::dtype)

    // constructor
    .def(py::init([](std::vector<double> data, std::vector<size_t> shape, size_t bidx, size_t pidx) {
        return new GradTensor(data, shape, bidx, pidx);
      }))
    .def(py::init([](std::vector<size_t> shape, size_t bidx, size_t pidx) {
        return new GradTensor(shape, bidx, pidx);
      }))
    .def(py::init([](double scalar) {
        return new GradTensor(scalar);
      }))
    .def_static("eye", &GradTensor::eye, 
        py::arg("n"), 
        py::arg("bidx"), 
        py::arg("pidx")
      )
    .def_static("gaussian", &GradTensor::gaussian, 
        py::arg("shape") = std::vector<size_t>{1, 1}, 
        py::arg("mean") = 0.0, 
        py::arg("stddev") = 1.0,
        py::arg("bidx") = 0,
        py::arg("pidx") = 1
      )
    .def_static("gaussian_like", &GradTensor::gaussian_like, 
        py::arg("input"), 
        py::arg("mean") = 0.0, 
        py::arg("stddev") = 1.0
      )
    .def_static("uniform", &GradTensor::uniform, 
        py::arg("shape") = std::vector<size_t>{1, 1}, 
        py::arg("min") = 0.0, 
        py::arg("max") = 1.0,
        py::arg("bidx") = 0,
        py::arg("pidx") = 1
      )
    .def_static("uniform_like", &GradTensor::uniform_like, 
        py::arg("input"), 
        py::arg("min") = 0.0, 
        py::arg("max") = 1.0
      )
    .def_static("ones", &GradTensor::ones, 
        py::arg("shape") = std::vector<size_t>{1, 1},
        py::arg("bidx") = 0,
        py::arg("pidx") = 1
      )
    .def_static("ones_like", &GradTensor::ones_like, 
        py::arg("input")
      )
    .def_static("zeros", &GradTensor::zeros, 
        py::arg("shape") = std::vector<size_t>{1, 1},
        py::arg("bidx") = 0,  
        py::arg("pidx") = 1 
      )
    .def_static("zeros_like", &GradTensor::zeros_like, 
        py::arg("input")
      )

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
    .def("shallowcopy", &GradTensor::shallowcopy)
    .def("deepcopy", &GradTensor::deepcopy)
    .def("copy", &GradTensor::copy)
    .def("reshape", 
        static_cast<GradTensor* (GradTensor::*)(std::vector<size_t>, bool)>(&GradTensor::reshape),
        py::arg("shape"), 
        py::arg("inplace") = false 
      )
    .def("transpose", 
        static_cast<GradTensor* (GradTensor::*)(const std::vector<size_t>&)>(&GradTensor::transpose)
      )
    .def("transpose", 
        static_cast<GradTensor* (GradTensor::*)(size_t, size_t)>(&GradTensor::transpose)
      )
    .def("transpose", 
        static_cast<GradTensor* (GradTensor::*)()>(&GradTensor::transpose)
      )

    // algebra
    .def("__neg__", [](GradTensor *a) { double c = -1.; return *a * c; })

    .def("__add__", [](GradTensor *a, Tensor *b) { return *a + b; })
    .def("__radd__", [](GradTensor *a, Tensor *b) { return *b + a; })
    .def("__add__", [](GradTensor *a, GradTensor *b) { return *a + b; })
    .def("__radd__", [](GradTensor *a, GradTensor *b) { return *b + a; })
    .def("__add__", [](GradTensor *a, double b) { return *a + b; })
    .def("__radd__", [](GradTensor *a, double b) { return *a + b; })

    .def("__sub__", [](GradTensor *a, Tensor *b) { return *a - b; })
    .def("__rsub__", [](GradTensor *a, Tensor *b) { return *b - a; })
    .def("__sub__", [](GradTensor *a, GradTensor *b) { return *a - b; })
    .def("__rsub__", [](GradTensor *a, GradTensor *b) { return *b - a; })
    .def("__sub__", [](GradTensor *a, double b) { return *a - b; })

    .def("__mul__", [](GradTensor *a, Tensor *b) { return *a * b; })
    .def("__rmul__", [](GradTensor *a, Tensor *b) { return *b * a; })
    .def("__mul__", [](GradTensor *a, GradTensor *b) { return *a * b; })
    .def("__rmul__", [](GradTensor *a, GradTensor *b) { return *b * a; })
    .def("__mul__", [](GradTensor *a, double b) { return *a * b; })
    .def("__rmul__", [](GradTensor *a, double b) { return *a * b; })

    .def("matmul", [](GradTensor* a, GradTensor* b) { return a->matmul(b); })
    .def("__matmul__", [](GradTensor* a, GradTensor* b) { return a->matmul(b); })

    // math
    .def("batchsum", &GradTensor::batchsum)
;}
