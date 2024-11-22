#include "common.hpp"

PYBIND11_MODULE(aten, m) {
    init_basetensor_binding(m);
    init_gradtensor_binding(m);
    init_tensor_binding(m);
}
