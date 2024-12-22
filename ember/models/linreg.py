import ember
from .regression import Regression

class LinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = ember.Tensor.ones([input_dim])
    self.b = ember.Tensor.ones([1])

  def forward(self, X: ember.Tensor): # B x D 
    self.z1 = X.dot(self.W) # B x D, D => B x 1
    self.z = self.z1 + self.b # B x 1, 1 => B x 1 
    return self.z

class BayesianLinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = ember.Tensor.gaussian([input_dim, 1], 0, 1)
    self.b = ember.Tensor.gaussian([20, 1], 0, 1)

  def forward(self, X: ember.Tensor): 
    self.z1 = X @ self.W
    z = self.z1 + self.b
    return z

