# Repository 

I've thought for a few weeks on how to structure this whole library, getting inspiration from the pytorch and tinygrad repositories. At a high level, the actual package repository is in `pyember/ember`, which uses functions pybinded from `pyember/aten` for fast computations. 

I tried to model a lot of the structure from Pytorch and TinyGrad. Very briefly, 

1. `aten/` contains the header and source files for the C++ low-level tensor library, such as basic operations and an autograd engine. 
 1. `aten/src` contains all the source files and definitions. 
 2. `aten/bindings` contains the pybindings. 
 3. `aten/test` contains all the C++ testing modules for aten. 
2. `ember/` contains the actual library, supporting high level models, objectives, optimizers, dataloaders, and samplers. 
 1. `ember/aten` contains the stub files. 
 2. `ember/datasets` contains all preprocessing tools, such as datasets/loaders, standardizing, cross validation checks. 
 3. `ember/models` contains all machine learning models. 
 4. `ember/objectives` contain all loss functions and regularizers. 
 5. `ember/optimizers` contain all the optimizers/solvers, such as iterative (e.g. SGD), greedy (e.g. decision tree splitting), and one-shot (e.g. least-squares solution). 
 6. `ember/samplers` contain all samplers (e.g. MCMC, SGLD). 
3. `docs/` contains detailed documentation about each function.  
4. `examples/` are example python scripts on training models.  
5. `tests/` are python testing modules for the `ember` library. 
6. `docker/` contains docker images of all the operating systems and architectures I tested ember on. General workflows on setting up the environment can be found there for supported machines. 
7. `setup.py` allows you to pip install this as a package. 
8. `run_tests.sh` which is the main test running script. 

For a more detailed explanation, look [here](docs/structure.md). 


## ATen

  Aten, short for "a tensor" library (got the name from PyTorch), is a C++ library that provides low level functionality for Tensors. This includes the basic vector and matrix operations like addition, scalar/matrix multiplication, dot products, transpose, etc, which are used everywhere in model training and inference and must be fast. 

### Compiling and PyBinding

  Let's look at `aten/CMakeLists.txt` and `aten/binding/CMakeLists.txt`. 

  - `aten/CMakeLists.txt` contains the instructions to generate a Makefile for compiling and linking the `aten` library. It has an optional argument `BUILD_PYTHON_BINDINGS` when set `ON`, will generate the `.so` file through `aten/binding/CMakeLists.txt`. The executables compiled with `aten/main.cpp` are compiled to `aten/build/main`. Same for the test files which are compiled to `aten/build/tests`. 

  - `aten/binding/CMakeLists.txt` contains the instructions to generate the `.so` file and saves it to `pyember/ember/_C.cpython-312-darwin.so`. It must be contained within the Python package directory, since `ember`cannot access libraries outside of its base directory. 
