# 🔥 Ember

Ember is a lightweight statistics and ML library for my personal use with C++ and Python. 

- [Installation](#installation) 
- [Getting Started](#getting-started)
  - [Ember Tensors](#ember-tensors)
  - [Datasets](#datasets)
  - [Models and Training](#models-and-training)
  - [Monte Carlo Samplers](#monte-carlo-samplers)

## Progress 

  To do: 
  1. Add a template argument for Tensor dtype. 
  2. Store all tensors in heap to preserve them after stack is destroyed.  

  ✅ - Done
  ❌ - Not implemented
  🚧 - Should not be accessed 
  🔨 - In progress 

  ### Aten BaseTensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Docs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ✅ | ✅ | ✅ |    
  | `std:: string dtype() const` | `dtype()` | ✅ | ✅ | ✅ | ✅ |    
  | `bool operator==(BaseTensor&)` | `__eq__()` | ✅ | ✅ | ✅ | ✅ |    
  | `bool operator!=(BaseTensor&)` | `__ne__()` | ✅ | ✅ | ✅ | ✅ |    
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` | ✅ | ✅ | ✅ | ✅ | 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` | ✅ | ✅ | ✅ | ✅ |    
  | `std::unique_ptr<BaseTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` | ✅ | ✅ | ✅ | ✅ | 

  ### Aten GradTensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Docs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ✅  | 
  | `std:: string dtype() const` | `dtype()` | ✅ | ✅ | 
  | `bool operator==(GradTensor&)` | `__eq__()` | ✅ | ✅ | 
  | `bool operator!=(GradTensor&)` | `__ne__()` | ✅ | ✅ | 
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` | ✅ | ✅ 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` | ✅ | ✅ 
  | `std::unique_ptr<GradTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` | ✅ | ✅ 
  | `pivot()` | `pivot()` | ✅  
  | `transpose()` |  `transpose()` | ✅    
  | `GradTensor copy() const` | `copy()` | ✅  
  | `GradTensor add(GradTensor&)` | `__add__(GradTensor)` | ✅ 
  | `Tensor add(Tensor&)` | `__add__(Tensor)` | ✅ 
  | `GradTensor add(ScalarTensor&)` | `__add__(ScalarTensor)` | ✅   
  | `GradTensor add(double&)` | `__add__(float)` | ✅   
  | `GradTensor sub(GradTensor&)` | `__sub__(GradTensor)` | ✅   
  | `Tensor sub(Tensor&)` | `__sub__(Tensor)` | ✅ 
  | `GradTensor sub(ScalarTensor&)` | `__sub__(ScalarTensor)` | ✅   
  | `GradTensor sub(double&)` | `__sub__(float)` | ✅   
  | `GradTensor mul(GradTensor&)` | `__mul__(GradTensor)` | ✅ 
  | `Tensor mul(Tensor&)` | `__mul__(Tensor)` | ✅ 
  | `GradTensor mul(ScalarTensor&)` | `__mul__(ScalarTensor)` | ✅   
  | `GradTensor mul(double&)` | `__mul__(float)` | ✅ 
  | `GradTensor matmul(GradTensor&)` | `__matmul__(GradTensor)` | ✅ 


  ### Aten Tensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Docs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ✅ | 
  | `std:: string dtype() const` | `dtype()` | ✅ | ✅ | 
  | `bool operator==(Tensor&)` | `__eq__()` | ✅ | ✅ | 
  | `bool operator!=(Tensor&)` | `__ne__()` | ✅ | ✅ | 
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` |  | ✅ 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` |  | ✅ 
  | `std::unique_ptr<Tensor> slice(const std::vector<Slice>&) const` | `__getitem__()` |  | ✅  
  | `Tensor(std::vector<double>, std::vector<size_t>)` | `Tensor(List[float], List[int])` | ✅   
  | `Tensor(std::vector<double>)` | `Tensor(List[float])` | ✅   
  | `Tensor(std::vector<std::vector<double>>)` | `Tensor(List[List[float]])` | ✅   
  | `Tensor(std::vector<std::vector<std::vector<double>>>);` | `Tensor(List[List[List[float]]])` | ✅   
  | `static Tensor arange(int, int, int)` | `Tensor.arange(int, int, int)` | ✅   
  | `static Tensor linspace(double, double, int)` | `Tensor.arange(float, float, int)` | ✅   
  | `static Tensor gaussian(std::vector<size_t> , double, double)` | `Tensor.arange(List[int], float, float)` | ✅   
  | `static Tensor uniform(std::vector<size_t> , double, double)` | `Tensor.uniform(List[int], int, int)` | ✅   
  | `static Tensor ones(std::vector<size_t>)` | `Tensor.ones(List[int])` | ✅   
  | `static Tensor zeros(std::vector<size_t>)` | `Tensor.zeros(List[int])` | ✅   
  | `void build_topo(Tensor* v, std::set<Tensor*>&, std::vector<Tensor*>&)` | 🚧 | ✅   
  | `std::vector<Tensor*> backprop(bool)` | `backprop(bool)` | ✅   
  | `Tensor copy() const` | `copy()` | ✅  
  | `Tensor add(Tensor&)` | `__add__(Tensor)` | ✅ 
  | `Tensor add(GradTensor&)` | `__add__(GradTensor)` | ✅ 
  | `Tensor add(ScalarTensor&)` | `__add__(ScalarTensor)` | ✅ 
  | `Tensor add(double&)` | `__add__(double)` | ✅ 
  | `Tensor sub(Tensor&)` |  | ✅ 
  | `Tensor sub(GradTensor&)`  |  | ✅
  | `Tensor sub(ScalarTensor&)` |  | ✅
  | `Tensor sub(double&)` |  | ✅
  | `Tensor mul(Tensor&)` |  | ✅
  | `Tensor mul(GradTensor&)` |  | ✅
  | `Tensor mul(ScalarTensor&)` |  | ✅
  | `Tensor mul(double&)` |  | ✅
  | `Tensor exp(ScalarTensor&, double&)` | ❌
  | `Tensor exp(double&, double&)` | ❌
  | `Tensor log(ScalarTensor&, double&)` | ❌
  | `Tensor log(double&, double&)` | ❌
  | `Tensor matmul(Tensor&)` |  | ✅
  | `Tensor sum(Tensor&)` | 🔨 
  | `Tensor tranpose(const std::vector<size_t>&) const` |  | ✅ 
  | `Tensor concat(Tensor&, size_t)` | ❌
  | `Tensor sin()` |  | ❌
  | `Tensor cos()` |  | ❌
  | `Tensor tan()` |  | ❌
  | `Tensor arcsin()` |  | ❌
  | `Tensor arccos()` |  | ❌
  | `Tensor arctan()` |  | ❌
  | `Tensor relu()` |  | ❌
  | `Tensor sigmoid()` |  | ❌
  | `Tensor leaky_relu()` |  | ❌

  ### ScalarTensor

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Docs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` |  | ✅  | 
  | `std:: string dtype() const` | `dtype()` |  | ✅ | 
  | `bool operator==(ScalarTensor&)` | `__eq__()` |  | ✅ | 
  | `bool operator!=(ScalarTensor&)` | `__ne__()` |  | ✅ | 
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` |  | ✅ 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` |  | ✅ 
  | `std::unique_ptr<ScalarTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` |  | ✅   
  | `ScalarTensor copy() const` |  | ✅
  | `double item() const` |  | ✅
  | `Tensor add(Tensor&)` |  | ✅
  | `GradTensor add(GradTensor&)` |  | ✅
  | `Tensor sub(Tensor&)` |  | ✅
  | `GradTensor sub(GradTensor&)` |  | ✅
  | `Tensor mul(Tensor&)` |  | ✅
  | `GradTensor mul(GradTensor&)` |  | ✅


  ### Models 


## Installation

  ### Compiling the `aten` Library  

  Git clone the repo, then pip install editable, which will run `setup.py`. I've painstakingly modified the Makefiles to make the installation as portable and easy as possible. You need Python 3.12 for now. 

  ```git clone git@github.com:mbahng/pyember.git 
  cd pyember 
  pip install -e .
  ```

  This runs `cmake` on `CMakeLists.txt`, which calls `aten/CMakeLists.txt` that compiles and links the source files in the C++ tensor library, which in turn (by default argument `BUILD_PYTHON_BINDINGS=ON`) calls `aten/bindings/CMakeLists.txt` to further generate a `.so` file that can be imported into `ember`. If there are problems, you should first check 

  1. Whether `build/` has been created. This is the first step in `setup.py` 
  2. Whether the compiled `main.cpp` and C++ unit test files have been compiled, i.e. if `aten/main` and `aten/tests` executables exist. 
  3. Whether `build/lib*/ember/aten.cpython-3**-darwin.so` exists. The Makefile generated by `aten/bindings/CMakeLists.txt` will produce `build/aten.cpython-3**-darwin.so`, which will immediately be moved by `setup.py` to `build/lib*/ember/aten.cpython-3**-darwin.so`. 
  4. The `setup()` function will copy this `.so` file to `ember/aten.cpython-3**-darwin.so`. The `.so` file must live within `ember`, the actual library, since `ember/__init__.py` must access it within the same directory level (cannot be higher). 

### Making sure everything's correct 

  Run the script `./run_tests.sh all` (args `python` to run just python tests and `cpp` to run just C++ tests), which will 
  1. Run all C++ unit tests for `aten`, ensuring that all functions work correctly. 
  2. Run all Python unit tests for `ember`, ensuring that additional functions work correctly and that the C++ functions are binded correctly. The stub (`.pyi`) files for `aten` are located in `ember/aten`. 

## Repository Structure 

  I tried to model a lot of the structure from Pytorch and TinyGrad.  

  1. `aten` contains the header and source files for the C++ low-level tensor library, such as basic operations and an autograd engine. 
  2. `docs` contains detailed documentation about each function.  
  3. `ember` contains the actual library, supporting high level models, dataloaders, and samplers. 
  4. `examples` are example scripts.  
  5. `tests` are testing modules for the `ember` library. 
  6. `CMakeLists.txt` generates the Makefiles needed to compile this library. 
  7. `setup.py` allows you to pip install this as a package. 
  8. `run_tests.sh` which is the main test running script. 

## Getting Started 

### Ember Tensors 

Tensors are multidimensional arrays that can be initialized in a number of ways. 
```
import ember 

a = ember.Tensor([2]) # scalar
b = ember.Tensor([1, 2, 3])  # vector 
c = ember.Tensor([[1, 2], [3, 4]]) # 2D vector 
d = ember.Tensor([[[1, 2]]]) # 3D vector
```
Gradient computations aren't supported for 3D vectors yet. Say that you have a series of elementary operations on tensors. 
```
a = ember.Tensor([2, -3]) 
h = a ** 2
b = ember.Tensor([3, 5])

c = b * h

d = ember.Tensor([10, 1])
e = c.dot(d) 

f = ember.Tensor([-2])

g = f * e
```

The C++ backend computes a directed acyclic graph (DAG) representing the operations done to compute `g`. You can then run `g.backwards()` to compute the gradients by applying the chain rule. This constructs the DAG and returns a topological sorting of its nodes. The gradients themselves, which are technically Jacobian matrices, are updated, with each mapping `x -> y` constructing a gradient tensor on `x` with value `dy/dx`. For now, the gradients are not accumulated for flexibility.  

```
top_sort = g.backprop()
print(a.grad) # [[4.0, 0.0], [0.0, -6.0]]
print(h.grad) # [[3.0, 0.0], [0.0, 5.0]]
print(b.grad) # [[4.0, 0.0], [0.0, 9.0]]
print(c.grad) # [[10.0, 1.0]]
print(d.grad) # [[12.0, 45.0]]
print(e.grad) # [[-2.0]]
print(f.grad) # [[165.0]]
print(g.grad) # [[1.0]]
```

This works for matrix multiplication as well, although in the backend the matrices are flattened and the 4-tensor is stored as a matrix. 
```
a = ember.Tensor([[1, 2, 3], [4, 5, 6]]) # 2 x 3
b =  ember.Tensor([[1, 2], [4, 8]]) # 2 x 2
c = b.matmul(a)

top_sort = c.backprop() 

pprint(a.grad)
[[1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
 [4.0, 0.0, 1.0, 8.0, 0.0, 2.0],
 [0.0, 4.0, 0.0, 0.0, 8.0, 0.0],
 [0.0, 0.0, 4.0, 0.0, 0.0, 8.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

pprint(b.grad)
[[1.0, 4.0, 0.0, 0.0],
 [2.0, 5.0, 0.0, 0.0],
 [3.0, 6.0, 1.0, 4.0],
 [0.0, 0.0, 2.0, 5.0],
 [0.0, 0.0, 3.0, 6.0],
 [0.0, 0.0, 0.0, 0.0]]
```

Finally, we can visualize this using the `networkx` package. 

![Alt text](docs/img/computational_graph.png)

### Datasets

### Models and Training

### Monte Carlo Samplers

## Contributing 

To implement a new functionality in the `aten` library, you must 
1. Add the class or function header in `aten/src/Tensor.h` 
2. Add the implementation in the correct file (or create a new one) in `aten./*Tensor/*.cpp`. Make sure to update `aten/bindings/CMakeLists.txt` if needed.
3. Add its pybindings (if a public function that will be used in `ember`) in `aten/bindings/*bindings.cpp`. Make sure to update `aten/bindings/CMakeLists.txt` if needed. 
4. Add relevant C++ tests in `aten/test/`.  
5. Not necessary, but it's good to test it out on a personal script for a sanity check.  
6. Add to the stub files in `ember/aten/*.pyi`. 
7. Add Python tests in `test/`. 
8. If everything passes, you can submit a pull request. 

