# 🔥 Ember

Ember is a statistics and ML library for my personal use with C++ and Python. I mainly built it for educational purposes, but it's quite functional and can be used to train several datasets. 

- [Installation](#installation) 
  - [Compilation](#compiling-the-aten-library)
  - [Testing](#testing) 
- [Repository Structure](#repository-structure) 
- [Getting Started](#getting-started)
  - [Ember Tensors and GradTensors](#ember-tensors-and-gradtensors)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Linear Regression](#linear-regression)
  - [Neural Nets](#multilayer-perceptrons)
  - [Datasets](#datasets)
  - [Models and Training](#models-and-training)
  - [Monte Carlo Samplers](#monte-carlo-samplers)

Look [here](docs/progress.md) to see the methods it supports. 

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

### Testing 

  Run the script `./run_tests.sh all` (args `python` to run just python tests and `cpp` to run just C++ tests), which will 
  1. Run all C++ unit tests for `aten`, ensuring that all functions work correctly. 
  2. Run all Python unit tests for `ember`, ensuring that additional functions work correctly and that the C++ functions are binded correctly. The stub (`.pyi`) files for `aten` are located in `ember/aten`. 

## Repository Structure 

  I tried to model a lot of the structure from Pytorch and TinyGrad. Very briefly, 

  1. `aten` contains the header and source files for the C++ low-level tensor library, such as basic operations and an autograd engine. 
  2. `docs` contains detailed documentation about each function.  
  3. `ember` contains the actual library, supporting high level models, dataloaders, and samplers. 
  4. `examples` are example scripts.  
  5. `tests` are testing modules for the `ember` library. 
  6. `CMakeLists.txt` generates the Makefiles needed to compile this library. 
  7. `setup.py` allows you to pip install this as a package. 
  8. `run_tests.sh` which is the main test running script. 

  For a more detailed explanation, look [here](docs/structure.md). 

## Getting Started 

### Ember Tensors and GradTensors

`ember.Tensor`s represent data and parameters, while `ember.GradTensor`s represent gradients. An advantage of this package is that rather than just supporting batch vector operations and matrix multiplications, we can also perform general contractions of rank $(N, M)$-tensors, a generalization of matrix multiplication. This allows us to represent and utilize the full power of higher order derivatives for arbitrary functions $f: \mathbb{R}^{\mathbf{M}} \rightarrow \mathbb{R}^{\mathbf{N}}$, where $\mathbf{M} = (M_1, \ldots, M_m)$ and $\mathbf{N} = (N_1, \ldots, N_m)$ are *vectors*, not just scalars, representing the dimension of each space. 

Tensors are multidimensional arrays that can be initialized in a number of ways. GradTensors are initialized during the backpropagation method, but we can explicitly set them if desired. 
```
import ember 

a = ember.Tensor([2]) # scalar
b = ember.Tensor([1, 2, 3])  # vector 
c = ember.Tensor([[1, 2], [3, 4]]) # 2D vector 
d = ember.Tensor([[[1, 2]]]) # 3D vector
```
Say that you have a series of elementary operations on tensors. 
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

### Automatic Differentiation

The C++ backend computes a directed acyclic graph (DAG) representing the operations done to compute `g`. You can then run `g.backprop()` to compute the gradients by applying the chain rule. This constructs the DAG and returns a topological sorting of its nodes. The gradients themselves, which are technically Jacobian matrices, are updated, with each mapping `x -> y` constructing a gradient tensor on `x` with value `dy/dx`. The gradients can be either accumulated by setting `backprop(intermediate=False)` so that the chain rule is not applied yet, or we can set `=True` to apply the chain rule to calculate the derivative of the tensor we called backprop on w.r.t. the rest of the tensors. 

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


Finally, we can visualize this using the `networkx` package. 

![Alt text](docs/img/computational_graph.png)

### Linear Regression 

To perform linear regression, use the `LinearRegression` model. 
```
import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(15) 
mse = ember.objectives.MSELoss()

for epoch in range(500): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop()
    model.step(1e-5) 

  print(loss)
``` 

### K Nearest Neighbors 

To do a simple K Nearest Neighbors regressor, use the following model. The forward method scans over the whole dataset, so we must input it to the model during instantiation. Note that we do not need a dataloader or a backpropagation method since we aren't iteratively updating gradients, though we want to show the loss. 

```
import ember
from ember.models import KNearestRegressor
from ember.datasets import LinearDataset

ds = LinearDataset(N=20, D=3)
model = KNearestRegressor(dataset=ds, K=1)
mse = ember.objectives.MSELoss() 

for k in range(1, 21): # hyperparameter tuning
  model.K = k
  print(f"{k} ===") 
  loss = 0
  for i in range(len(ds)): 
    x, y = ds[i] 
    y_ = model.forward(x) 
    loss = loss + mse(y, y_) 

  print(loss)
```

### Multilayer Perceptrons 

To instantiate a MLP, just call it from models. In here we make a 2-layer MLP with a dummy dataset. For now only SGD with batch size 1 is supported.  
```
import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.MultiLayerPerceptron(15, 10) 
mse = ember.objectives.MSELoss()

for epoch in range(500):  
  loss = None
  for x, y in dl: 
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop() 
    model.step(1e-5)

  print(loss)
```
Its outputs over 1 minute. 
```
LOSS = 256733.64437981808
LOSS = 203239.08846901066
LOSS = 160223.4554735339
LOSS = 125704.33716141782
LOSS = 98074.96981384761
LOSS = 76026.19871949886
LOSS = 58491.92389906721
LOSS = 44604.493032865605
LOSS = 33658.23285350788
LOSS = 25079.638682869212
LOSS = 18403.01062298029
LOSS = 13250.54496118543
LOSS = 9316.069468116035
LOSS = 6351.758695807299
LOSS = 4157.286052245369
LOSS = 2570.96819208677
LOSS = 1462.5380952427417
LOSS = 727.2493587808174
LOSS = 281.0683664354656
LOSS = 56.75530418715159
```

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

