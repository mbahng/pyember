from ember import Tensor, ScalarTensor
from .regression import Regression
from .model import track_access
from typing import Union

class LinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = Tensor.ones([input_dim])
    self.b = Tensor.ones([1])
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    self.z1 = X.dot(self.W) # B x D, D => B x 1 
    self.z = self.z1 + self.b # B x 1, 1 => B x 1 

    super().forward(X)
    return self.z

  def step(self, a: Union[ScalarTensor, float, int]) -> None: 
    self.W -= a * self.W.grad.batchsum().reshape(self.W.shape)
    self.b -= a * self.b.grad.batchsum().reshape(self.b.shape)

class BayesianLinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = Tensor.gaussian([input_dim, 1], 0, 1)
    self.b = Tensor.gaussian([20, 1], 0, 1)

  def forward(self, X: Tensor): 
    self.z1 = X @ self.W
    z = self.z1 + self.b
    return z

