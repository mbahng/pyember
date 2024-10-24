#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "tensor.h" 

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {

  py::class_<Tensor>(m, "Tensor") 

    // Constructors
      .def(py::init([](std::vector<double> data) {
        return Tensor(data);
      }))
      
    // Attributes
      .def_readwrite("data", &Tensor::data)
      .def_readwrite("grad", &Tensor::grad)
      .def_readwrite("prev", &Tensor::prev)
      .def("backward", 
        [](Tensor &t) {
          if (t.backward) {
              t.backward();
          }
        }
      )

    .def("__repr__",
      [](Tensor &a) {
        return static_cast<std::string>(a);
      })
    
    .def("__str__",
      [](Tensor &a) {
        return static_cast<std::string>(a);
      })

    .def("__add__", 
        [](Tensor &a, Tensor &b) {
            return a.add(b);
        }
      )

    .def("__mul__", 
        [](Tensor &a, Tensor &b) {
            return a.mult(b);
        }
      )

    .def("backprop", 
        [](Tensor &a) {
            return a.backprop(); 
        }
      )

    .def("dot", 
        [](Tensor &a, Tensor &b) {
            return a.dot(b);
        }
      ); 

}
