#include "common.hpp"

void init_gradtensor_binding(py::module_ &m) {
  py::class_<GradTensor, BaseTensor>(m, "GradTensor")
    .def(py::init<>());
}


