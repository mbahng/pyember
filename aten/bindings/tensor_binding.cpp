#include "common.hpp"

void init_tensor_binding(py::module_ &m) {
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


