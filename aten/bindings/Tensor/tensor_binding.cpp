#include "../common.hpp"
#include "../../src/Tensor/Tensor.h"

void init_tensor_binding(py::module_ &m) {
  // Bind the Tensor class
  py::class_<Tensor, BaseTensor>(m, "Tensor")

    // base
    .def_property_readonly("prev", &Tensor::prev)
    .def_readwrite("requires_grad", &Tensor::requires_grad)
    .def_readwrite("grad", &Tensor::grad)
    .def_property_readonly("type", &Tensor::type)
    .def_property_readonly("dtype", &Tensor::dtype)
    .def("backprop", &Tensor::backprop, 
        py::arg("intermediate") = false 
    )

    // constructor
    .def(py::init([](double scalar, bool requires_grad = true) {
        return new Tensor(scalar, requires_grad); 
        }), 
        py::arg("scalar"), 
        py::arg("requires_grad") = true
      )
    .def(py::init([](std::vector<double> storage, std::vector<size_t> shape, size_t bidx = 0, bool requires_grad = true) {
        return new Tensor(storage, shape, bidx, requires_grad);
        }), 
        py::arg("storage"), 
        py::arg("shape"), 
        py::arg("bidx") = 0, 
        py::arg("requires_grad") = true
      )
    .def(py::init([](std::vector<double> storage, size_t bidx = 0,bool requires_grad = true) {
        return new Tensor(storage, bidx, requires_grad);
        }),
        py::arg("storage"), 
        py::arg("bidx") = 0, 
        py::arg("requires_grad") = true
      )
    .def(py::init([](std::vector<std::vector<double>> storage, size_t bidx = 0,bool requires_grad = true) {
        return new Tensor(storage, bidx, requires_grad);
        }),
        py::arg("storage"), 
        py::arg("bidx") = 0, 
        py::arg("requires_grad") = true
      )
    .def(py::init([](std::vector<std::vector<std::vector<double>>> storage, size_t bidx = 0,bool requires_grad = true) {
        return new Tensor(storage, bidx, requires_grad);
        }),
        py::arg("storage"), 
        py::arg("bidx") = 0, 
        py::arg("requires_grad") = true
      )

    .def_static("arange", &Tensor::arange, 
        py::arg("start"), 
        py::arg("stop"), 
        py::arg("step") = 1, 
        py::arg("requires_grad") = true
      )
    .def_static("linspace", &Tensor::linspace,
        py::arg("start"), 
        py::arg("stop"), 
        py::arg("numsteps"),
        py::arg("requires_grad") = true
      )
    .def_static("gaussian", &Tensor::gaussian, 
        py::arg("shape") = std::vector<size_t>{1}, 
        py::arg("mean") = 0.0, 
        py::arg("stddev") = 1.0,
        py::arg("bidx") = 0,
        py::arg("requires_grad") = true
      )
    .def_static("uniform", &Tensor::uniform, 
        py::arg("shape") = std::vector<size_t>{1}, 
        py::arg("min") = 0.0, 
        py::arg("max") = 1.0,
        py::arg("bidx") = 0,
        py::arg("requires_grad") = true
      )
    .def_static("ones", &Tensor::ones, 
        py::arg("shape") = std::vector<size_t>{1},
        py::arg("bidx") = 0,
        py::arg("requires_grad") = true
      )
    .def_static("zeros", &Tensor::zeros, 
        py::arg("shape") = std::vector<size_t>{1},
        py::arg("bidx") = 0,  
        py::arg("requires_grad") = true
      )

    // string
    .def("__repr__", &Tensor::operator std::string, py::is_operator())
    .def("__str__", &Tensor::operator std::string, py::is_operator())
    .def("meta", &Tensor::meta)

    // comparison 
    .def("__eq__", &Tensor::operator==, py::is_operator())
    .def("__ne__", &Tensor::operator!=, py::is_operator()) 
    .def("__lt__", &Tensor::operator<, py::is_operator()) 
    .def("__gt__", &Tensor::operator>, py::is_operator()) 
    .def("__le__", &Tensor::operator<=, py::is_operator()) 
    .def("__ge__", &Tensor::operator>=, py::is_operator()) 

    // index 

    // shape 
    .def("shallowcopy", &Tensor::shallowcopy, py::arg("requires_grad") = true)
    .def("deepcopy", &Tensor::deepcopy, py::arg("requires_grad") = true)
    .def("copy", &Tensor::copy, py::arg("requires_grad") = true)
    .def("reshape", 
        [](Tensor &a, std::vector<size_t> newshape, bool inplace = true, bool requires_grad = true) {
            return a.reshape(newshape, inplace, requires_grad);
        }, 
        py::arg("shape"), 
        py::arg("inplace") = true, 
        py::arg("requires_grad") = true
      )
    .def("transpose", 
        [](Tensor &a, size_t d1, size_t d2, bool requires_grad = true) {
            return a.transpose(d1, d2, requires_grad);
        }, 
        py::arg("d1"), 
        py::arg("d2"), 
        py::arg("requires_grad") = true
      )
    .def("transpose", 
        [](Tensor &a, bool requires_grad = true) {
            return a.transpose(requires_grad);
        }, 
        py::arg("requires_grad") = true
      )
    .def("transpose", 
        [](Tensor &a, const std::vector<size_t> &axes, bool requires_grad = true) {
            return a.transpose(axes, requires_grad);
        }, 
        py::arg("axes") = std::vector<size_t>{1, 0}, 
        py::arg("requires_grad") = true
      )
    .def("T", 
        [](Tensor &a, size_t d1, size_t d2, bool requires_grad = true) {
            return a.transpose(d1, d2, requires_grad);
        }, 
        py::arg("d1"), 
        py::arg("d2"), 
        py::arg("requires_grad") = true
      )
    .def("T", 
        [](Tensor &a, bool requires_grad = true) {
            return a.transpose(requires_grad);
        }, 
        py::arg("requires_grad") = true
      )
    .def("T", 
        [](Tensor &a, const std::vector<size_t> &axes, bool requires_grad = true) {
            return a.transpose(axes, requires_grad);
        },
        py::arg("axes") = std::vector<size_t>{1, 0},
        py::arg("requires_grad") = true
      )

    // algebra 
    .def("__neg__", [](Tensor *a) { double c = -1.; return *a * c; ; })

    .def("__add__", [](Tensor *a, Tensor *b) { return *a + b; })
    .def("__radd__", [](Tensor *a, Tensor *b) { return *b + a; })
    .def("__add__", [](Tensor *a, GradTensor *b) { return *a + b; })
    .def("__radd__", [](Tensor *a, GradTensor *b) { return *b + a; })
    .def("__add__", [](Tensor *a, double b) { return *a + b; })
    .def("__radd__", [](Tensor *a, double b) { return *a + b; })

    .def("__iadd__", [](Tensor *a, GradTensor *b) { return *a += b; })

    .def("__sub__",  [](Tensor *a, Tensor *b) { return *a - b; })
    .def("__rsub__", [](Tensor *a, Tensor *b) { return *b - a; })
    .def("__sub__", [](Tensor *a, GradTensor *b) { return *a - b; })
    .def("__rsub__", [](Tensor *a, GradTensor *b) { return *b - a; })
    .def("__sub__", [](Tensor *a, double b) { return *a - b; })
    // rsub needs to be implemented for doubles

    .def("__isub__", [](Tensor *a, GradTensor *b) { return *a -= b; })

    .def("__mul__", [](Tensor *a, Tensor *b) { return *a * b; })
    .def("__rmul__", [](Tensor *a, Tensor *b) { return *b * a; })
    .def("__mul__", [](Tensor *a, GradTensor *b) { return *a * b; })
    .def("__rmul__", [](Tensor *a, GradTensor *b) { return *b * a; })
    .def("__mul__", [](Tensor *a, double b) { return *a * b; })
    .def("__rmul__", [](Tensor *a, double b) { return *a * b; })

    .def("__imul__", [](Tensor *a, GradTensor *b) { return *a *= b; })

    .def("matmul", [](Tensor *a, Tensor *b) { return a->matmul(b);  })
    .def("__matmul__", [](Tensor *a, Tensor *b) { return a->matmul(b);  })

    // math
    .def("dot", [](Tensor *a, Tensor*b) { return a->dot(b); })
    .def("sum", [](Tensor *a) { return a->sum(); })
    .def("sum", [](Tensor *a, size_t dim) { return a->sum(dim); })
    .def("sum", [](Tensor *a, std::vector<size_t> dim) { return a->sum(dim); })
    .def("__pow__", [](Tensor *a, double* x) { return a->pow(x); })
    .def("exp", [](Tensor *a, double* x) { return a->pow(x); })
    .def("relu", [](Tensor *a) { return a->relu(); })

;}
