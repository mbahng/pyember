#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../src/Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(aten, m) {
  // First bind the BaseTensor class since Tensor inherits from it
  py::class_<BaseTensor>(m, "BaseTensor")
    .def_property("storage",
      [](const BaseTensor &t) -> const std::vector<double>& { return t.storage_; },
      [](BaseTensor &t, const std::vector<double> &value) { t.storage_ = value; })
    .def_property("shape",
      [](const BaseTensor &t) -> const std::vector<size_t>& { return t.shape_; },
      [](BaseTensor &t, const std::vector<size_t> &value) { t.shape_ = value; });

  // Bind the GradTensor minimally since it's used by Tensor
  py::class_<GradTensor, BaseTensor>(m, "GradTensor")
    .def(py::init<>());

  // Bind the Tensor class
  py::class_<Tensor, BaseTensor>(m, "Tensor")
    // Constructor from data and shape
    .def(py::init<std::vector<double>, std::vector<size_t>>())
    // Expose grad attribute
    .def_property("grad",
      [](const Tensor &t) -> const GradTensor& { return t.grad; },
      [](Tensor &t, const GradTensor &g) { t.grad = g; })
    // Expose prev vector
    .def_property("prev",
      [](const Tensor &t) -> const std::vector<Tensor*>& { return t.prev; },
      [](Tensor &t, const std::vector<Tensor*> &p) { t.prev = p; })
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
      });
}
