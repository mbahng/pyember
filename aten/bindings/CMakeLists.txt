set(BINDING_SOURCES
    aten_binding.cpp
    Tensor/gradtensor_binding.cpp
    Tensor/basetensor_binding.cpp 
    Tensor/tensor_binding.cpp 
)
# Get Python extension suffix more robustly
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(aten_python ${BINDING_SOURCES})
target_link_libraries(aten_python PRIVATE aten)
target_include_directories(aten_python 
  PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(aten_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "aten"
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)
