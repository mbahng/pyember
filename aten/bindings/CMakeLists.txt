cmake_minimum_required(VERSION 3.10)

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python: ${Python_VERSION}")


set(BINDING_SOURCES
    aten_binding.cpp
    gradtensor_binding.cpp
    basetensor_binding.cpp 
    tensor_binding.cpp 
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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    OUTPUT_NAME "aten"
)
