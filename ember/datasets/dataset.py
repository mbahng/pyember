from .. import Tensor 

class Dataset(): 

  def __init__(self, X: Tensor, Y: Tensor) -> None: 
    if (X.has_grad or Y.has_grad): 
      raise Exception("X or Y has gradients and will be backpropagated. Did you mean to do this?")

    self.index = 0
    self.X = X 
    self.Y = Y 
    self.D = X.shape[1:] 
    self.X.bidx = 1
    self.Y.bidx = 1

    assert self.X.shape[0] == self.Y.shape[0], \
      f"Length of input ({self.X.shape[0]}) does not \
       match length of output ({self.Y.shape[0]})."

    self.limit = self.X.shape[0]

  def __len__(self): 
    return self.limit 

  def __getitem__(self, i: int):  
    if i < self.limit: 
      return (
        self.X[i,:].reshape(self.X[i,:].nbshape), 
        self.Y[i,:].reshape(self.Y[i,:].nbshape)
      )
    else: 
      raise Exception("Indices are out of bound. ")

  def __iter__(self): 
    return self 

  def __next__(self): 
    if self.index >= self.limit: 
      self.index = 0
      raise StopIteration 
    # take off the batch index, should have a squeeze method 
    pair = (
      self.X[self.index,:].reshape(self.X[self.index,:].nbshape),
      self.Y[self.index,:].reshape(self.Y[self.index,:].nbshape)
    )
    self.index += 1 # self.index is an int so passed by value 
    return pair

class LinearDataset(Dataset): 

  def __init__(self, N, D, noise = "gaussian") -> None: 
    """
    Generates a dummy dataset with targets linear w.r.t. covariates, w/ noise.  
    """
    X = Tensor.uniform([N, D + 1], 0, 10, has_grad = False) # +1 for bias term 
    for i in range(N): 
      X[i, 0] = 1.0 
    self.truth_param = Tensor.uniform([D + 1, 1], -10, 10, has_grad = False)
    Y_truth = X @ self.truth_param
    noise = Tensor.gaussian([N, 1], 0, 1, has_grad=False)
    Y = Y_truth + noise

    super().__init__(X, Y)
  
class QuadraticDataset(Dataset): 

  def __init__(self, X: Tensor, Y: Tensor) -> None: 
    super().__init__(X, Y)
  
class CubicDataset(Dataset): 

  def __init__(self, X: Tensor, Y: Tensor) -> None: 
    super().__init__(X, Y)
  


