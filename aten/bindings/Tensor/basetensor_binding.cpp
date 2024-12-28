#include "../common.hpp"
#include "../../src/Tensor/Tensor.h"

void init_basetensor_binding(py::module_ &m) {
  py::class_<Slice>(m, "Slice")
    .def(py::init<size_t, size_t, size_t>(),
       py::arg("start") = 0,
       py::arg("stop") = std::numeric_limits<size_t>::max(),
       py::arg("step") = 1)
    .def_readwrite("start", &Slice::start)
    .def_readwrite("stop", &Slice::stop)
    .def_readwrite("step", &Slice::step);

  py::class_<BaseTensor>(m, "BaseTensor")

    // base
    .def_property_readonly("storage", &BaseTensor::storage)
    .def_property_readonly("rank", &BaseTensor::rank)
    .def_property_readonly("hdim", &BaseTensor::hdim)
    .def_property_readonly("shape", &BaseTensor::shape)
    .def_property_readonly("bshape", &BaseTensor::bshape)
    .def_property_readonly("nbshape", &BaseTensor::nbshape)
    .def_readwrite("bidx", &BaseTensor::bidx)
    .def_property_readonly("type", &BaseTensor::type)
    .def_property_readonly("dtype", &BaseTensor::dtype)
    .def("is_scalar", &BaseTensor::is_scalar)
    .def("item", &BaseTensor::item)
    .def("__len__", &BaseTensor::size)
    .def("__float__", &BaseTensor::item)
    .def("__int__", [](BaseTensor &t) -> int64_t {
        return static_cast<int64_t>(t.item());
    })

    // string
    .def("__repr__", &BaseTensor::operator std::string, py::is_operator()) 
    .def("__str__", &BaseTensor::operator std::string, py::is_operator())
    .def("meta", &BaseTensor::meta)

    // comparison
    .def("__eq__", &BaseTensor::operator==, py::is_operator())
    .def("__ne__", &BaseTensor::operator!=, py::is_operator()) 

    // index
    .def("at", py::overload_cast<const std::vector<size_t>&>(&BaseTensor::at))
    .def("at", py::overload_cast<const std::vector<size_t>&>(&BaseTensor::at, py::const_))
    
    .def("__index__", [](BaseTensor &t) -> int64_t {
        return static_cast<int64_t>(t.item());
    })

    .def("__getitem__", [](BaseTensor &t, const py::object &index) -> py::object {
      if (py::isinstance<py::tuple>(index)) {
        // Handle multiple indices
        py::tuple idx_tuple = index.cast<py::tuple>();
        std::vector<Slice> slices;
        
        // First, convert all indices to appropriate slices
        for (size_t i = 0; i < t.shape().size(); ++i) {
          if (i < idx_tuple.size()) {
            const auto& idx = idx_tuple[i];
            if (py::isinstance<py::slice>(idx)) {
              py::slice slice = idx.cast<py::slice>();
              size_t start, stop, step, slicelength;
              if (!slice.compute(t.shape()[i], &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
              }
              slices.push_back(Slice(start, stop, step));
            } else {
              // Convert integer index to single-element slice
              size_t idx_val = idx.cast<size_t>();
              slices.push_back(Slice(idx_val, idx_val + 1, 1));
            }
          } else {
            // If fewer indices than dimensions, use full slice for remaining dimensions
            slices.push_back(Slice(0, t.shape()[i], 1));
          }
        }

        auto out = t.slice(slices); 
        // All indices are now converted to slices
        // Can't do py::cast(out) directly for some reason
        return py::cast(t.slice(slices)); 
      } else if (py::isinstance<py::int_>(index)) {
        // Single integer index - convert to slice
        size_t idx_val = index.cast<size_t>();
        std::vector<Slice> slices;
        slices.push_back(Slice(idx_val, idx_val + 1, 1));
        // Add full slices for remaining dimensions
        for (size_t i = 1; i < t.shape().size(); ++i) {
          slices.push_back(Slice(0, t.shape()[i], 1));
        }
        std::unique_ptr<BaseTensor> out = t.slice(slices); 
        return py::cast(t.slice(slices)); 
      } else if (py::isinstance<py::slice>(index)) {
        // Single slice
        py::slice slice = index.cast<py::slice>();
        size_t start, stop, step, slicelength;
        if (!slice.compute(t.shape()[0], &start, &stop, &step, &slicelength)) {
          throw py::error_already_set();
        }
        std::vector<Slice> slices;
        slices.push_back(Slice(start, stop, step));
        // Add full slices for remaining dimensions
        for (size_t i = 1; i < t.shape().size(); ++i) {
          slices.push_back(Slice(0, t.shape()[i], 1));
        }
        std::unique_ptr<BaseTensor> out = t.slice(slices); 
        return py::cast(t.slice(slices)); 
      }
      throw py::type_error("Invalid index type");
    })

    .def("__setitem__", [](BaseTensor &t, const py::object &index, double value) {
      if (py::isinstance<py::tuple>(index)) {
        py::tuple idx_tuple = index.cast<py::tuple>();
        std::vector<size_t> indices;
        for (const auto& idx : idx_tuple) {
          indices.push_back(idx.cast<size_t>());
        }
        t.at(indices) = value;
      } else if (py::isinstance<py::int_>(index)) {
        t.at(std::vector<size_t>{index.cast<size_t>()}) = value;
      } else {
        throw py::type_error("Invalid index type");
      }
    });

    
;}
