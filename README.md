# 🔥 Ember

Ember is a lightweight statistics and ML library for my personal use with C++ and Python. 

- [Installation](#installation) 
- [Getting Started](#getting-started)
  - [Ember Tensors](#ember-tensors)
  - [Datasets](#datasets)
  - [Models and Training](#models-and-training)
  - [Monte Carlo Samplers](#monte-carlo-samplers)

## Installation

Once you git pull, copy the `pybind11` repo into `_CPP` directory. 
```
cd ember/_CPP 
git submodule add https://github.com/pybind/pybind11.git
git submodule update --init
```
Build the `_CPP` library for low level tensor operations with the following. Make sure `cmake` is installed. 
``` 
cd ember/_CPP 
mkdir build && cd build 
cmake ..
make 
```
Copy the `.so` file to `ember`. Then it is accessible by the python modules. Finally, run `pip install -e .` in the root directory to install the package. 

## Getting Started 

### Ember Tensors 

### Datasets

### Models and Training

### Monte Carlo Samplers

