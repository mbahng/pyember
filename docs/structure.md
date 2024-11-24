# Repository 

I've thought for a few weeks on how to structure this whole library, getting inspiration from the pytorch and tinygrad repositories. At a high level, the actual package repository is in `pyember/ember`, which uses functions pybinded from `pyember/aten` for fast computations. 

## ATen

  Aten, short for "a tensor" library (stole the name from PyTorch), is a C++ library that provides low level functionality for Tensors. This includes the basic vector and matrix operations like addition, scalar/matrix multiplication, dot products, transpose, etc, which are used everywhere in model training and inference and must be fast. 

### Compiling and PyBinding

  Let's look at `aten/CMakeLists.txt` and `aten/binding/CMakeLists.txt`. You must have Python 3.12.

  - `aten/CMakeLists.txt` contains the instructions to generate a Makefile for compiling and linking the `aten` library. It has an optional argument `BUILD_PYTHON_BINDINGS` when set `ON`, will generate the `.so` file through `aten/binding/CMakeLists.txt`. The executables compiled with `aten/main.cpp` are compiled to `aten/build/main`. Same for the test files which are compiled to `aten/build/tests`. 

  - `aten/binding/CMakeLists.txt` contains the instructions to generate the `.so` file and saves it to `pyember/ember/_C.cpython-312-darwin.so`. It must be contained within the Python package directory, since `ember`cannot access libraries outside of its base directory. 


### Structure

  1. The `aten/src` directory contains all the source files and implementations for all `Tensor`s. The main header file, `Tensor.h` contains the declarations of the Tensor classes (more on it here) and their methods.
  2. The `aten/test` directory contains all the C++ unit tests to ensure that all methods are working as expected. It uses Google Test (`gtest`). 

