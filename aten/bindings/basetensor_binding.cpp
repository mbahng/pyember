#include "common.hpp"

void init_basetensor_binding(py::module_ &m) {
  py::class_<BaseTensor::Slice>(m, "Slice")
    .def(py::init<size_t, size_t, size_t>(),
       py::arg("start") = 0,
       py::arg("stop") = std::numeric_limits<size_t>::max(),
       py::arg("step") = 1)
    .def_readwrite("start", &BaseTensor::Slice::start)
    .def_readwrite("stop", &BaseTensor::Slice::stop)
    .def_readwrite("step", &BaseTensor::Slice::step);

  py::class_<BaseTensor>(m, "BaseTensor")
    .def_property("storage",
        [](const BaseTensor &t) -> const std::vector<double>& { 
          return t.storage_; 
        },
        [](BaseTensor &t, const std::vector<double> &value) { 
          t.storage_ = value; 
        }
      )
    .def_property("shape",
        [](const BaseTensor &t) -> const std::vector<size_t>& { 
          return t.shape_; 
        },
        [](BaseTensor &t, const std::vector<size_t> &value) { 
        t.shape_ = value; 
        }
      )
    .def_property("bshape",
        [](const BaseTensor &t) -> const std::vector<size_t>& { 
          return t.bshape_; 
        },
        [](BaseTensor &t, const std::vector<size_t> &value) { 
          t.bshape_ = value; 
        }
      )
    .def_property("nbshape",
        [](const BaseTensor &t) -> const std::vector<size_t>& { 
          return t.nbshape_; 
        },
        [](BaseTensor &t, const std::vector<size_t> &value) {
          t.nbshape_ = value; 
        }
      )
    .def_property("bidx",
        [](const BaseTensor &t) -> const size_t { 
          return t.bidx_ ; 
        },
        [](BaseTensor &t, const size_t &value) { 
          t.bidx_ = value;  
          t.bshape_ = std::vector<size_t>(t.shape_.begin(), t.shape_.begin() + t.bidx_);
          t.nbshape_ = std::vector<size_t>(t.shape_.begin() + t.bidx_, t.shape_.end());
        }
      )

    .def("__repr__", &BaseTensor::operator std::string, py::is_operator()) 
    .def("__str__", &BaseTensor::operator std::string, py::is_operator())
    .def("__eq__", &BaseTensor::operator==, py::is_operator())
    .def("__ne__", &BaseTensor::operator!=, py::is_operator()) 
    .def("type", &BaseTensor::type, py::is_operator())
    .def("dtype", &BaseTensor::dtype, py::is_operator())
    .def("data", &BaseTensor::data, py::is_operator())
    .def("size", &BaseTensor::shape, py::is_operator())
    .def("at", py::overload_cast<const std::vector<size_t>&>(&BaseTensor::at))
    .def("at", py::overload_cast<const std::vector<size_t>&>(&BaseTensor::at, py::const_))

    .def("meta", 
        [](BaseTensor *a) {
          a->meta(); 
        }
      )
    .def("__len__", 
        [](Tensor *a) {
          return (a->data()).size();
        }
      )

    .def("__getitem__", [](BaseTensor &t, const py::object &index) -> py::object {
      if (py::isinstance<py::tuple>(index)) {
        // Handle multiple indices
        py::tuple idx_tuple = index.cast<py::tuple>();
        std::vector<BaseTensor::Slice> slices;
        
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
              slices.push_back(BaseTensor::Slice(start, stop, step));
            } else {
              // Convert integer index to single-element slice
              size_t idx_val = idx.cast<size_t>();
              slices.push_back(BaseTensor::Slice(idx_val, idx_val + 1, 1));
            }
          } else {
            // If fewer indices than dimensions, use full slice for remaining dimensions
            slices.push_back(BaseTensor::Slice(0, t.shape()[i], 1));
          }
        }

        auto out = t.slice(slices); 
        if ((out->data()).size() == 1) {
          return py::cast(ScalarTensor(out->data()));
        }
        
        // All indices are now converted to slices
        // Can't do py::cast(out) directly for some reason
        return py::cast(t.slice(slices)); 
      } else if (py::isinstance<py::int_>(index)) {
        // Single integer index - convert to slice
        size_t idx_val = index.cast<size_t>();
        std::vector<BaseTensor::Slice> slices;
        slices.push_back(BaseTensor::Slice(idx_val, idx_val + 1, 1));
        // Add full slices for remaining dimensions
        for (size_t i = 1; i < t.shape().size(); ++i) {
          slices.push_back(BaseTensor::Slice(0, t.shape()[i], 1));
        }
        std::unique_ptr<BaseTensor> out = t.slice(slices); 
        if ((out->data()).size() == 1) {
          return py::cast(ScalarTensor(out->data()));
        }
        return py::cast(t.slice(slices)); 
      } else if (py::isinstance<py::slice>(index)) {
        // Single slice
        py::slice slice = index.cast<py::slice>();
        size_t start, stop, step, slicelength;
        if (!slice.compute(t.shape()[0], &start, &stop, &step, &slicelength)) {
          throw py::error_already_set();
        }
        std::vector<BaseTensor::Slice> slices;
        slices.push_back(BaseTensor::Slice(start, stop, step));
        // Add full slices for remaining dimensions
        for (size_t i = 1; i < t.shape().size(); ++i) {
          slices.push_back(BaseTensor::Slice(0, t.shape()[i], 1));
        }
        std::unique_ptr<BaseTensor> out = t.slice(slices); 
        if ((out->data()).size() == 1) {
          return py::cast(ScalarTensor(out->data()));
        }
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
