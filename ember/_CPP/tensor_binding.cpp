#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "tensor.h" 

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {

  py::class_<GradTensor>(m, "GradTensor")

    // Attributes
      .def_readwrite("data", &GradTensor::data)
      .def_readwrite("shape", &GradTensor::shape)
      .def_readwrite("length", &GradTensor::length)
    
    // Constrctors 
      .def(py::init([](std::vector<double> data, std::vector<int> shape) {
        return GradTensor(data, shape); 
      }))

      .def(py::init([](std::vector<std::vector<double>> data) {
        return GradTensor(data); 
      }))

    // Utility Functions
      .def("__repr__", [](GradTensor &a) { return static_cast<std::string>(a); })
      .def("__str__", [](GradTensor &a) { return static_cast<std::string>(a); })
      .def("reshape", &GradTensor::reshape)

    // Math Operations
    .def("matmul", 
       [](GradTensor &a, GradTensor &b) {
         return a.matmul(b); 
     })

    ;

  py::class_<Tensor>(m, "Tensor") 

    // Attributes
      .def_readwrite("data", &Tensor::data)
      .def_readwrite("shape", &Tensor::shape)
      .def_readwrite("length", &Tensor::length)
      .def_readwrite("grad", &Tensor::grad)
      .def_readwrite("prev", &Tensor::prev, py::keep_alive<1, 2>())

    // Constructors
      .def(py::init([](std::vector<double> data, std::vector<int> shape) {
        return Tensor(data, shape);
      }))
      
      .def(py::init([](std::vector<double> data) {
        return Tensor(data);
      }))

      .def(py::init([](std::vector<std::vector<double>> data) {
        return Tensor(data); 
      }))

      .def(py::init([](std::vector<std::vector<std::vector<double>>> data) {
        return Tensor(data); 
      }))

      .def_static("gaussian", &Tensor::gaussian,
        py::arg("shape"),
        py::arg("mean") = 0.0,
        py::arg("stddev") = 1.0
      )

      .def_static("uniform", &Tensor::uniform,
        py::arg("shape"),
        py::arg("min") = 0.0,
        py::arg("max") = 1.0
      )

      .def_static("ones", &Tensor::ones,
        py::arg("shape")
      )

      .def_static("zeros", &Tensor::zeros,
        py::arg("shape")
      )

    // Utility Functions 
      .def("__eq__", &Tensor::operator==, py::is_operator())
      .def("__ne__", &Tensor::operator!=, py::is_operator())

      .def("copy", &Tensor::copy)

      .def("__repr__", [](Tensor &a) { return static_cast<std::string>(a); })
      .def("__str__", [](Tensor &a) { return static_cast<std::string>(a); })

      .def("__len__", [](Tensor &a) { return a.length; })
      .def("reshape", &Tensor::reshape)

    // Indexing and Slicing 
      .def("__getitem__", [](const Tensor& self, py::object index) {
        if (py::isinstance<py::tuple>(index)) {
          std::vector<Tensor::Slice> slices;
          for (auto item : index) {
            if (py::isinstance<py::slice>(item)) {
              py::slice slice = item.cast<py::slice>();
              py::ssize_t start, stop, step, slicelength;
              if (!slice.compute(self.shape[slices.size()], &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
              }
              slices.emplace_back(start, stop, step);
            } else {
              int idx = item.cast<int>();
              slices.emplace_back(idx, idx + 1, 1);
            }
          }
          return self.slice(slices);
        } 
        else if (py::isinstance<py::slice>(index)) {
          py::slice slice = index.cast<py::slice>();
          py::ssize_t start, stop, step, slicelength;
          if (!slice.compute(self.shape[0], &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
          }
          return self.slice({Tensor::Slice(start, stop, step)});
        } 
        else {
          // For single integer index, return a tensor with the indexed value
          std::vector<size_t> indices = {index.cast<size_t>()};
          return Tensor(std::vector<double>{self.at(indices)}); 
        }
          })

      .def("__setitem__", [](Tensor& self, py::object index, double value) {
        if (py::isinstance<py::tuple>(index)) {
          std::vector<size_t> indices;
          for (auto item : index.cast<py::tuple>()) {
            indices.push_back(item.cast<size_t>());
          }
          self.at(indices) = value;
        } else {
          self.at({index.cast<size_t>()}) = value;
        }
      })

    // Backpropagation Functions 
      .def("backward", 
        [](Tensor &t) {
          if (t.backward) {
              t.backward();
          } 
        }
      )

      .def("backprop", 
        [](Tensor &a) {
            return a.backprop(); 
        }
      )
    
    // Unary Argument Scalar Math Operations 
      .def("sum", 
         [](Tensor &a) {
           return a.sum(); 
       })

      .def("mean", 
         [](Tensor &a) {
           return a.mean(); 
       })

    // Unary Argument Vector Math Operations 
      .def("__pow__", 
          [](Tensor &a, double n) {
              return a.pow(n);
          }
        )

    // Binary Argument Scalar Math Operations 
      .def("dot", 
          [](Tensor &a, Tensor &b) {
              return a.dot(b);
          }
        )

    // Binary Argument Vector Math Operations 
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

      .def("__mul__", 
          [](Tensor &a, Tensor &b) {
              return a.mult(b);
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

    // Activation Functions
      .def("relu", 
         [](Tensor &a) {
           return a.relu();
         })

  ;
}
