#include "common.hpp"

void init_tensor_binding(py::module_ &m) {
  // Bind the Tensor class
  py::class_<Tensor, BaseTensor>(m, "Tensor")

    // Constructors
    .def(py::init([](std::vector<double> data, std::vector<size_t> shape) {
        return new Tensor(data, shape);
      }))
    .def(py::init([](std::vector<double> data) {
        return new Tensor(data);
      }))
    .def(py::init([](std::vector<std::vector<double>> data) {
        return new Tensor(data);
      }))
    .def(py::init([](std::vector<std::vector<std::vector<double>>> data) {
        return new Tensor(data);
      }))

    .def_static("arange", &Tensor::arange, 
      py::arg("start"), py::arg("stop"), py::arg("step"))
    .def_static("linspace", &Tensor::linspace, 
      py::arg("start"), py::arg("stop"), py::arg("numsteps"))
    .def_static("gaussian", &Tensor::gaussian, 
      py::arg("shape"), py::arg("mean"), py::arg("stddev"))
    .def_static("uniform", &Tensor::uniform, 
      py::arg("shape"), py::arg("min"), py::arg("max"))
    .def_static("ones", &Tensor::ones, 
      py::arg("shape"))
    .def_static("zeros", &Tensor::zeros, 
      py::arg("shape"))
    .def("copy", 
        [](Tensor &a) {
          return a.copy();
        }
      )

    // Expose grad attribute
    .def_property("grad",
      [](const Tensor &t) -> const GradTensor* { return t.grad; },
      [](Tensor &t, GradTensor *g) { t.grad = g; })

    // Expose prev vector
    .def_property("prev",
      [](const Tensor &t) -> const std::vector<Tensor*>& { return t.prev; },
      [](Tensor &t, const std::vector<Tensor*> &p) { t.prev = p; }) 

    // Backpropagation Functions 
      .def("backward", 
        [](Tensor &t) {
          if (t.backward) {
              t.backward();
          } 
        }
      )

      .def("backprop", &Tensor::backprop, 
          py::arg("intermediate") = false 
      )

    .def("reshape", 
        [](Tensor &a, std::vector<size_t> newshape, bool inplace = true) {
            return a.reshape(newshape, inplace);
        }, 
        py::arg("shape"), 
        py::arg("inplace") = true
      )
    .def("transpose", 
        [](Tensor &a, const std::vector<size_t> &axes) {
            return a.transpose();
        }
      )
    // order of the bindings matter: most specific should go first
    .def("__neg__", 
        [](Tensor *a) {
          double c = -1.;
          return a->mul(&c);
        }
      )
    .def("__add__", 
        [](Tensor *a, ScalarTensor *b) {
          return a->add(b);
        }
      )
    .def("__radd__", 
        [](Tensor *a, ScalarTensor *b) {
          return b->add(a);
        }
      )
    .def("__add__", 
        [](Tensor *a, Tensor *b) {
          return a->add(b);
        }
      )
    .def("__radd__", 
        [](Tensor *a, Tensor *b) {
          return b->add(a);
        }
      )
    .def("__add__", 
        [](Tensor *a, GradTensor *b) {
          return a->add(b);
        }
      )
    .def("__radd__", 
        [](Tensor *a, GradTensor *b) {
          return b->add(a);
        }
      )
    .def("__add__", 
        [](Tensor *a, double *b) {
          return a->add(b);
        }
      )
    .def("__radd__", 
        [](Tensor *a, double *b) {
          return a->add(b);
        }
      )

    .def("__sub__", 
        [](Tensor *a, ScalarTensor *b) {
          return a->sub(b);
        }
      )
    .def("__rsub__", 
        [](Tensor *a, ScalarTensor *b) {
          return b->sub(a);
        }
      )
    .def("__sub__", 
        [](Tensor *a, Tensor *b) {
          return a->sub(b);
        }
      )
    .def("__rsub__", 
        [](Tensor *a, Tensor *b) {
          return b->sub(a);
        }
      )
    .def("__sub__", 
        [](Tensor *a, GradTensor *b) {
          return a->sub(b);
        }
      )
    .def("__rsub__", 
        [](Tensor *a, GradTensor *b) {
          return b->sub(a);
        }
      )
    .def("__sub__", 
        [](Tensor *a, double *b) {
          return a->sub(b);
        }
      )
    .def("__rsub__", 
        [](Tensor *a, double *b) {
          ScalarTensor* scalar = new ScalarTensor(*b); 
          return scalar->sub(a);
        }
      )

    .def("__mul__", 
        [](Tensor *a, ScalarTensor *b) {
          return a->mul(b);
        }
      )
    .def("__rmul__", 
        [](Tensor *a, ScalarTensor *b) {
          return b->mul(a);
        }
      )
    .def("__mul__", 
        [](Tensor *a, Tensor *b) {
          return a->mul(b);
        }
      )
    .def("__rmul__", 
        [](Tensor *a, Tensor *b) {
          return b->mul(a);
        }
      )
    .def("__mul__", 
        [](Tensor *a, GradTensor *b) {
          return a->mul(b);
        }
      )
    .def("__rmul__", 
        [](Tensor *a, GradTensor *b) {
          return b->mul(a);
        }
      )
    .def("__mul__", 
        [](Tensor *a, double *b) {
          return a->mul(b);
        }
      )
    .def("__rmul__", 
        [](Tensor *a, double *b) {
          return a->mul(b);
        }
      )

    .def("matmul", 
       [](Tensor *a, Tensor *b) {
         return a->matmul(b); 
       }
     )
    .def("__matmul__", 
       [](Tensor *a, Tensor *b) {
         return a->matmul(b); 
       }
     )

    .def("sum", 
       [](Tensor *a) {
         return a->sum();
       }
     )
    .def("__pow__", 
       [](Tensor *a, double* x) {
         return a->pow(x);
       }
     )
    .def("exp", 
       [](Tensor *a, double* x) {
         return a->pow(x);
       }
     )
    .def("relu", 
       [](Tensor *a) {
         return a->relu();
       }
     )

;}
