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

    .def_static("arange", 
        [](int start, int stop, int step = 1) {
            return Tensor::arange(start, stop, step);
        }, 
        py::arg("start"), 
        py::arg("stop"), 
        py::arg("step") = 1
      )
    .def_static("linspace", 
        [](double start, double stop, int numsteps) {
            return Tensor::linspace(start, stop, numsteps);
        }, 
        py::arg("start"), 
        py::arg("stop"), 
        py::arg("numsteps")
      )
    .def_static("gaussian", 
        [](std::vector<size_t> shape = {1}, double mean = 0.0, double stddev = 1.0) {
            return Tensor::gaussian(shape, mean, stddev);
        }, 
        py::arg("shape") = std::vector<size_t>{1}, 
        py::arg("mean") = 0.0, 
        py::arg("stddev") = 1.0
      )
    .def_static("uniform", 
        [](std::vector<size_t> shape = {1}, double min = 0.0, double max = 1.0) {
            return Tensor::uniform(shape, min, max);
        }, 
        py::arg("shape") = std::vector<size_t>{1}, 
        py::arg("min") = 0.0, 
        py::arg("max") = 1.0
      )
    .def_static("ones", 
        [](std::vector<size_t> shape = {1}) {
            return Tensor::ones(shape);
        }, 
        py::arg("shape") = std::vector<size_t>{1}
      )
    .def_static("zeros", 
        [](std::vector<size_t> shape = {1}) {
            return Tensor::zeros(shape);
        }, 
        py::arg("shape") = std::vector<size_t>{1}
      )

    .def("copy", 
        [](Tensor &a) {
          return a.copy();
        }
      )

    // Expose grad attribute
    .def_property("grad",
      [](const Tensor &t) -> const GradTensor* { 
        if (t.grad == nullptr) {
          throw std::logic_error("Gradient is not initialized. Call backprop(). ");
        }
        return t.grad; 
      },
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
        }, 
        py::arg("axes") = std::vector<size_t>{0, 1}
      )
    .def("T", 
        [](Tensor &a, const std::vector<size_t> &axes) {
            return a.transpose();
        },
        py::arg("axes") = std::vector<size_t>{0, 1}
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
