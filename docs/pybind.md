# Compiling and Pybinding 

The C++ library is called `aten`, short for "a tensor" and modeled off of the original PyTorch repository. 

Let's look at `aten/CMakeLists.txt` and `aten/binding/CMakeLists.txt`. You must have Python 3.12.

- `aten/CMakeLists.txt` contains the instructions to generate a Makefile for compiling and linking the `aten` library. It has an optional argument `BUILD_PYTHON_BINDINGS` when set `ON`, will generate the `.so` file through `aten/binding/CMakeLists.txt`. The executables compiled with `aten/main.cpp` are compiled to `aten/build/main`. Same for the test files which are compiled to `aten/build/tests`. 

- `aten/binding/CMakeLists.txt` contains the instructions to generate the `.so` file and saves it to `pyember/ember/_C.cpython-312-darwin.so`. It must be contained within the Python package directory, since `ember`cannot access libraries outside of its base directory. 




