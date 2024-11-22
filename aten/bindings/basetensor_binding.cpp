#include "common.hpp"

void init_basetensor_binding(py::module_ &m) {
  py::class_<BaseTensor>(m, "BaseTensor")
    .def_property("storage",
      [](const BaseTensor &t) -> const std::vector<double>& { return t.storage_; },
      [](BaseTensor &t, const std::vector<double> &value) { t.storage_ = value; })
    .def_property("shape",
      [](const BaseTensor &t) -> const std::vector<size_t>& { return t.shape_; },
      [](BaseTensor &t, const std::vector<size_t> &value) { t.shape_ = value; });
}

