#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h" 

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {

  py::class_<Tensor>(m, "Tensor") 

    // Constructors
      .def(py::init([](std::vector<double> input_data, std::vector<int> input_shape) {
        return Tensor(input_data, input_shape); 
      }))

      .def(py::init([](std::vector<double> input_data) {
        return Tensor(input_data); 
      }))

      .def(py::init([](std::vector<std::vector<double>> input_data) {
        return Tensor(input_data); 
      }))

      .def(py::init([](std::vector<std::vector<std::vector<double>>> input_data) {
        return Tensor(input_data); 
      }))

    // Attributes
      .def_readwrite("shape", &Tensor::shape)
      .def_readwrite("data", &Tensor::data)

    .def("__repr__",
      [](Tensor &a) {
        return static_cast<std::string>(a);
      })
    
    .def("__str__",
      [](Tensor &a) {
        return static_cast<std::string>(a);
      })
    
    .def("__eq__", &Tensor::operator==, py::is_operator())
    .def("__ne__", &Tensor::operator!=, py::is_operator())

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
        return Tensor(std::vector<double>{self.at(indices)}, std::vector<int>{1});
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

    .def("__add__", 
      [](Tensor &a , Tensor &b) {
        return a.add(b); 
      })
    
    .def("__add__", 
      [](Tensor &a , double &b) {
        return a.add(b); 
      })
    
    .def("__add__", 
      [](double &a , Tensor &b) {
        return b.add(a); 
      })
    
    .def("__sub__", 
      [](Tensor &a , Tensor &b) {
        return a.sub(b); 
      })
    
    .def("__sub__", 
      [](Tensor &a , double &b) {
        return a.sub(b); 
      })
    
    .def("__sub__", 
      [](double &a , Tensor &b) {
        // To do, fix this since it's not symmetric. 
        return b.sub(a); 
      })
    
    .def("__mul__", 
      [](Tensor &a , Tensor &b) {
        return a.mul(b); 
      })
    
    .def("__mul__", 
      [](Tensor &a , double b) {
        return a.mul(b); 
      })
    
    .def("__rmul__", 
      [](Tensor &a , double b) {
        return a.mul(b); 
      })

    .def("__pow__", 
      [](Tensor &a , double &b) {
        return a.pow(b); 
      })

    .def("sum", 
      [](Tensor &b) {
        return b.sum(); 
      })
    
    .def("mean", 
      [](Tensor &b) {
        return b.mean(); 
      })
    
    .def("norm", 
      [](Tensor &b) {
        return b.norm(); 
      })
    
    .def("dot", 
      [](Tensor &a, Tensor &b) {
        return a.dot(b); 
      })
    
    .def("matmul", 
       [](Tensor &a, Tensor &b) {
         return a.matmul(b); 
       })

    .def("__matmul__", 
       [](Tensor &a, Tensor &b) {
         return a.matmul(b); 
       })

    .def("reshape", &Tensor::reshape)

    .def("transpose", &Tensor::transpose, 
         py::arg("dim1") = 0, 
         py::arg("dim2") = 1)

    .def("T", &Tensor::transpose, 
         py::arg("dim1") = 0, 
         py::arg("dim2") = 1)
    ; 

  m.def("gaussian", &gaussian, 
        py::arg("shape"), 
        py::arg("mean") = 0.0, 
        py::arg("std_dev") = 1.0); 

  m.def("uniform", &uniform, 
        py::arg("shape"), 
        py::arg("min") = 0.0, 
        py::arg("max") = 1.0); 
}
