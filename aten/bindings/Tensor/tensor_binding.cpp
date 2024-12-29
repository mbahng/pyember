#include "../common.hpp"
#include "../../src/Tensor/Tensor.h"

void init_tensor_binding(py::module_ &m) {
  // Bind the Tensor class
  py::class_<Tensor, BaseTensor>(m, "Tensor")

    // base
    .def_property_readonly("prev", &Tensor::prev)
    .def_property_readonly("type", &Tensor::type)
    .def_property_readonly("dtype", &Tensor::dtype)
    .def_readwrite("grad", &Tensor::grad)
    /* .def_property("_grad", */
    /*   [](const Tensor &t) -> const GradTensor* {  */
    /*     if (t.grad == nullptr) { */
    /*       throw std::logic_error("Gradient is not initialized. Call backprop(). "); */
    /*     } */
    /*     return t.grad;  */
    /*   }, */
    /*   [](Tensor &t, GradTensor *g) { t.grad = g; }) */
    /*  */
    .def("backprop", &Tensor::backprop, 
        py::arg("intermediate") = false 
    )
    .def_readwrite("requires_grad", &Tensor::requires_grad)

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

    // comparison 

    // index 

    // shape 
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
        [](Tensor &a, const std::vector<size_t> &axes, bool inplace = false, bool requires_grad = true) {
            return a.transpose(axes, inplace, requires_grad);
        }, 
        py::arg("axes") = std::vector<size_t>{1, 0}, 
        py::arg("inplace") = false,
        py::arg("requires_grad") = true
      )
    .def("T", 
        [](Tensor &a, const std::vector<size_t> &axes, bool inplace = false, bool requires_grad = true) {
            return a.transpose(axes, inplace, requires_grad);
        },
        py::arg("axes") = std::vector<size_t>{1, 0},
        py::arg("inplace") = false,
        py::arg("requires_grad") = true
      )

    // algebra 
    .def("__neg__", [](Tensor *a) { return a->mul(-1.); })
    .def("__add__", [](Tensor *a, Tensor *b) { return a->add(b); })
    .def("__radd__", [](Tensor *a, Tensor *b) { return b->add(a); })
    .def("__add__", [](Tensor *a, GradTensor *b) { return a->add(b); })
    .def("__radd__", [](Tensor *a, GradTensor *b) { return b->add(a); })
    .def("__iadd__", [](Tensor *a, GradTensor *b) { return a->iadd(b); })
    .def("__add__", [](Tensor *a, double b) { return a->add(b); })
    .def("__radd__", [](Tensor *a, double b) { return a->add(b); })

    .def("__sub__",  [](Tensor *a, Tensor *b) { return a->sub(b); })
    .def("__rsub__", [](Tensor *a, Tensor *b) { return b->sub(a); })
    .def("__sub__", [](Tensor *a, GradTensor *b) { return a->sub(b); })
    .def("__rsub__", [](Tensor *a, GradTensor *b) { return b->sub(a); })
    .def("__isub__", [](Tensor *a, GradTensor *b) { return a->isub(b); })
    .def("__sub__", [](Tensor *a, double b) { return a->sub(b); })
    .def("__mul__", [](Tensor *a, Tensor *b) { return a->mul(b); })
    .def("__rmul__", [](Tensor *a, Tensor *b) { return b->mul(a); })
    .def("__mul__", [](Tensor *a, GradTensor *b) { return a->mul(b); })
    .def("__rmul__", [](Tensor *a, GradTensor *b) { return b->mul(a); })
    .def("__imul__", [](Tensor *a, GradTensor *b) { return a->imul(b); })
    .def("__mul__", [](Tensor *a, double b) { return a->mul(b); })
    .def("__rmul__", [](Tensor *a, double b) { return a->mul(b); })

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
