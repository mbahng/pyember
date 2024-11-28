import ember
from . import Model

class Regression(Model): 

  def __init__(self): 
    super().__init__()
    self.params = {}

class LinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = ember.Tensor.gaussian([input_dim, 1], 0, 1)
    self.b = ember.Tensor.gaussian([20, 1], 0, 1)

  def forward(self, X: ember.Tensor): 
    self.z1 = X @ self.W
    z = self.z1 + self.b
    return z


class MultiLayerPerceptron(Regression): 

  def __init__(self, input_dim: int, hidden_dim:int): 
    super().__init__() 
    self.W1 = ember.Tensor.gaussian([input_dim, hidden_dim], 0, 1) 
    self.b1 = ember.Tensor.gaussian([1, hidden_dim], 0, 1) 
    self.W2 = ember.Tensor.gaussian([hidden_dim, 1], 0, 1)
    self.b2 = ember.Tensor.gaussian([1, 1], 0, 1) 

  def forward(self, X: ember.Tensor): 
    self.W1a = X @ self.W1 
    self.b1a = self.W1a + self.b1
    self.o1a = self.b1a.relu() 
    self.W2a = self.o1a @ self.W2 
    self.b2a = self.W2a + self.b2 
    return self.b2a

