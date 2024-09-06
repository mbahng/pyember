#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"  // This should be the header file containing your Tensor class

namespace py = pybind11;

PYBIND11_MODULE(tensor_cpp, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<double>&>())
        .def("__len__", &Tensor::size)
        .def("__getitem__", [](const Tensor &t, size_t i) {
            if (i >= t.size()) throw py::index_error();
            return t[i];
        })
        .def("__setitem__", [](Tensor &t, size_t i, double v) {
            if (i >= t.size()) throw py::index_error();
            t[i] = v;
        })
        .def("__str__", &Tensor::str)
        .def("__repr__", &Tensor::str);
}

