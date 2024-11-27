import ember
from . import Model
from .. import Tensor

class Regression(Model): 

  def __init__(self): 
    super().__init__()
    self.params = {}

class LinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = ember.Tensor.gaussian([input_dim, 1], 0, 1)
    self.z1 = None
    self.b = ember.Tensor.gaussian([1, 1], 0, 1)

  def forward(self, X: Tensor): 
    self.z1 = X @ self.W
    z = self.z1 + self.b
    return z


class MultiLayerPerceptron(Regression): 

  def __init__(self, input_dim: int, hidden_dim:int): 
    super().__init__() 
    self.W1 = ember.Tensor.gaussian([hidden_dim, input_dim], 0, 1) 
    self.W1a = None
    self.b1 = ember.Tensor.gaussian([hidden_dim, 1], 0, 1) 
    self.b1a = None
    self.W2 = ember.Tensor.gaussian([1, hidden_dim], 0, 1)
    self.W2a = None
    self.b2 = ember.Tensor.gaussian([1, 1], 0, 1) 
    self.b2a = None

  def forward(self, X: Tensor): 
    self.W1a = self.W1 @ X 
    self.b1a = self.W1a + self.b1
    self.W2a = self.W2 @ self.b1a 
    self.b2a = self.W2a + self.b2 
    return self.b2a

