#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_basetensor_binding(py::module_ &);
void init_gradtensor_binding(py::module_ &);
void init_tensor_binding(py::module_ &);
