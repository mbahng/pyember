#include "common.hpp"

void init_tensor_binding(py::module_ &m) {
  // Bind the Tensor class
  py::class_<Tensor, BaseTensor>(m, "Tensor")

    // Constructors
    .def(py::init<std::vector<double>, std::vector<size_t>>(), 
        py::arg("data"), py::arg("shape"))
    .def(py::init<std::vector<double>>(), 
        py::arg("data")) 
    .def(py::init<std::vector<std::vector<double>>>(), 
        py::arg("data")) 
    .def(py::init<std::vector<std::vector<std::vector<double>>>>(), 
        py::arg("data")) 

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
      [](const Tensor &t) -> const GradTensor& { return t.grad; },
      [](Tensor &t, const GradTensor &g) { t.grad = g; })

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

    .def("transpose", 
        [](Tensor &a, const std::vector<size_t> &axes) {
            return a.transpose();
        }
      )

    // String representation
    .def("__repr__",
      [](const Tensor &t) {
        std::string shape_str = "(";
        for (size_t i = 0; i < t.shape_.size(); ++i) {
          if (i > 0) shape_str += ", ";
          shape_str += std::to_string(t.shape_[i]);
        }
        shape_str += ")";
        return "Tensor(shape=" + shape_str + ")";
      })

    .def("__add__", 
        [](Tensor &a, Tensor &b) {
            return a.add(b);
        }
      )

    .def("__add__", 
        [](Tensor &a, GradTensor &b) {
            return a.add(b);
        }
      )

    .def("__sub__", 
        [](Tensor &a, Tensor &b) {
            return a.sub(b);
        }
      )

    .def("__sub__", 
        [](Tensor &a, GradTensor &b) {
            return a.sub(b);
        }
      )

    .def("__mul__", 
        [](Tensor &a, Tensor &b) {
            return a.mul(b);
        }
      )
  
    .def("__mul__", 
        [](Tensor &a, GradTensor &b) {
            return a.mul(b);
        }
      )
  
    .def("matmul", 
       [](Tensor &a, Tensor &b) {
         return a.matmul(b); 
       })

    .def("__matmul__", 
       [](Tensor &a, Tensor &b) {
         return a.matmul(b); 
       })

;}


