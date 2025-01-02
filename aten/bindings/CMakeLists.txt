find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python: ${Python_VERSION}")
message(STATUS "Python Executable: ${Python_EXECUTABLE}")

find_package(pybind11 REQUIRED)

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

pybind11_add_module(aten_python_module ${BINDING_SOURCES})
target_link_libraries(aten_python_module PRIVATE aten)
target_include_directories(aten_python_module 
  PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(aten_python_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "aten"
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)

set(BINDING_SOURCES
    aten_binding.cpp
    Tensor/gradtensor_binding.cpp
    Tensor/basetensor_binding.cpp 
    Tensor/tensor_binding.cpp 
)

# Create the Python module named 'aten'
pybind11_add_module(aten_python ${BINDING_SOURCES})  

# Link with the C++ library
target_link_libraries(aten_python PRIVATE aten)

target_include_directories(aten_python 
  PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set output directory and rename the output file to 'aten.so'
set_target_properties(aten_python PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  OUTPUT_NAME "aten" 
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
