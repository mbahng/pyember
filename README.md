# 🔥 Ember

Ember is a statistics and ML library for my personal use with C++ and Python. I mainly built it for educational purposes, but it's quite functional and can be used to train several datasets. 

- [Installation](#installation) 
- [Ember Tensors and GradTensors](#ember-tensors-and-gradtensors)
    - [Ember Tensors](#tensors)
    - [GradTensors and Automatic Differentiation](#gradtensors-and-automatic-differentiation)
- [Data](#data)
    - [Datasets](#datasets) 
    - [Data Analysis and Feature Extraction](#data-analysis-and-feature-extraction) 
    - [Dataloaders](#dataloader) 
- [Models](#models)
    - [Linear Regression](#linear-regression)
    - [Neural Nets](#multilayer-perceptrons)
    - [Datasets](#datasets)
- [Objectives](#objectives)
- [Optimizers](#optimizers)
- [Monte Carlo Samplers](#monte-carlo-samplers)

From a statistical learning theory perspective, let's consider the essential building blocks for machine learning. 
1. We need to define a class of functions, aka our **model**. 
2. We need to when define some sort of metric to judge the performance of our model. This is done by defining an **objective**, which may or may not have a regularization term. 
3. We need to integrate through over our data generating probability measure, which measures our risk. Summing over our **dataset** allows us to compute the empirical risk. 
4. Using the empirical risk, in practice we must find the optimal model using some **optimizer** (e.g. SGD). 

This is generally how the library is structured, and generally, every training model will look something like this. 
```
import ember 

ds = ember.datasets.Dataset(...) 
dl = ember.datasets.Dataloader(ds, batch_size=b)  
model = ember.models.Model(...) 
objective = ember.objectives.Loss(...)

for epoch in range(500): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = objective(y, y_)
    loss.backprop()
    model.step(1e-5) 

  print(loss)
```


## Installation 

This package is published to [PyPI](https://pypi.org/project/pyember/). I recommend to first create a virtual environment with Python 3.9+ installed and run 
```
pip install pyember
``` 
It supports Linux AMD64, MacOS 11+, and Windows 11 out of the box. If you would like to build from source or find details on which machines it has been specifically tested on, look at the [installation details](docs/installation.md).  

## Ember Tensors and GradTensors

`ember.Tensor`s represent data and parameters, while `ember.GradTensor`s represent gradients. An advantage of this package is that rather than just supporting batch vector operations and matrix multiplications, we can also perform general contractions of rank $(N, M)$-tensors, a generalization of matrix multiplication. This allows us to represent and utilize the full power of higher order derivatives for arbitrary functions $f: \mathbb{R}^{\mathbf{M}} \rightarrow \mathbb{R}^{\mathbf{N}}$, where $\mathbf{M} = (M_1, \ldots, M_m)$ and $\mathbf{N} = (N_1, \ldots, N_m)$ are *vectors* (representing, through abuse of notation, the Cartesian product). 

### Tensors
Tensors are multidimensional arrays that can be initialized in a number of ways. GradTensors are initialized during the backpropagation method, but we can explicitly set them if desired. 
```
from ember import Tensor 

Tensor(scalar=2)
Tensor(storage=[2]) 

Tensor(storage=[1, 2, 3]) 
Tensor.arange(0, 10, 2)

Tensor(storage=[[1, 2], [3, 4]]) 
Tensor.gaussian([2, 2],mean=0, stddev=1)

Tensor([ [[1, 2], [3, 4]], [[5, 6], [7, 8]] ]) 
Tensor.uniform([2, 3, 4], min=0, max=1) 
Tensor.linspace(0, 100, 45).reshape([3, 3, 5])
```

### GradTensors and Automatic Differentiation

Say that you have a series of elementary operations on tensors (by default, with `requires_grad=True`). 
```
a = Tensor([2, -3]) # [2, -3]
h = a ** 2          # [4, 9]
b = Tensor([3, 5])  # [3, 5]

c = b * h           # [12, 45]

d = Tensor([10, 1]) # [10, 1]
e = c.dot(d)        # 165

f = Tensor(-2)      # -2

g = f * e           # -330
```

The C++ backend computes a directed acyclic graph (DAG) representing the operations done to compute `g`. You can then run `g.backprop()` to compute the gradients by applying the chain rule. This constructs the DAG and returns a topological sorting of its nodes. The gradients themselves, which are technically Jacobian matrices, are updated, with each mapping `x -> y` constructing a gradient tensor on `x` with value `dy/dx`. The gradients can be either accumulated by setting `backprop(intermediate=False)` so that the chain rule is not applied yet, or we can set `=True` to apply the chain rule to calculate the derivative of the tensor we called backprop on w.r.t. the rest of the tensors. 

```
top_sort = g.backprop() # a topologically sorted list of these GradTensor's 
print(a.grad) # [-240, +60]
print(h.grad) # [-60, -10]
print(b.grad) # [-80, -18]
print(c.grad) # [-20, -2]
print(d.grad) # [-24, -90]
print(e.grad) # [-2]
print(f.grad) # [165]
print(g.grad) # [1]
```

## Data

### Datasets 

### Data Analysis and Feature Extraction 

Support for pandas-like feature extraction models will be implemented. 

### Dataloaders 

## Models 

### Linear Regression 

To perform linear regression, use the `LinearRegression` model. 
```
import ember 

ds = ember.datasets.LinearDataset(N=20, D=15)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(15) 
mse = ember.objectives.MSELoss() 
optim = ember.optimizers.SGDOptimizer(model, 1e-4)

for epoch in range(1000): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop()
    optim.step()
  
  if epoch % 100 == 0: 
    print(loss)
``` 

### K Nearest Neighbors 

To do a simple K Nearest Neighbors regressor, use the following model. The forward method scans over the whole dataset, so we must input it to the model during instantiation. Note that we do not need a dataloader or a backpropagation method since we aren't iteratively updating gradients, though we want to show the loss. We simply evaluate this model over the hyperparameter $K$. 

```
import ember
from ember.models import KNearestRegressor
from ember.datasets import LinearDataset

ds = LinearDataset(N=20, D=3)
model = KNearestRegressor(dataset=ds, K=1)
mse = ember.objectives.MSELoss() 

for k in range(1, 21):
  model.K = k
  loss = ember.Tensor(0)
  for i in range(len(ds)): 
    x, y = ds[i] 
    y_ = model.forward(x) 
    loss = loss + mse(y, y_) 

  print(f"{k} : {float(loss)}")  # type: ignore
```

### Multilayer Perceptrons 

To instantiate a MLP, just call it from models. In here we make a 2-layer MLP with a dummy dataset. For now only SGD with batch size 1 is supported.  
```
import ember 

ds = ember.datasets.LinearDataset(N=20, D=15)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.MultiLayerPerceptron(15, 10) 
mse = ember.objectives.MSELoss()
optim = ember.optimizers.SGDOptimizer(model, 1e-5)

for epoch in range(500):  
  loss = None
  for x, y in dl: 
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop() 
    optim.step()

  if epoch % 25 == 0: 
    print(loss)
```

## Objectives

## Optimizers

## Monte Carlo Samplers
