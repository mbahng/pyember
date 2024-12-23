import ember
from .regression import Regression

class MultiLayerPerceptron(Regression): 

  def __init__(self, input_dim: int, hidden_dim:int): 
    super().__init__() 
    self.W1 = ember.Tensor.gaussian([input_dim, hidden_dim], 0, 1) 
    self.b1 = ember.Tensor.gaussian([hidden_dim], 0, 1) 
    self.W2 = ember.Tensor.gaussian([hidden_dim], 0, 1)
    self.b2 = ember.Tensor.gaussian([1], 0, 1) 
    self.intermediate = dict() 

  def forward(self, X: ember.Tensor): 
    self.intermediate["W1"] = self.W1
    self.W1a = X @ self.W1 
    self.intermediate["W1a"] = self.W1a
    self.intermediate["b1"] = self.b1
    self.b1a = self.W1a + self.b1
    self.intermediate["b1a"] = self.b1a
    self.o1a = self.b1a.relu() 
    self.intermediate["o1a"] = self.o1a
    self.intermediate["W2"] = self.W2
    self.W2a = self.o1a.dot(self.W2) 
    self.intermediate["W2a"] = self.W2a 
    self.intermediate["b2"] = self.b2
    self.b2a = self.W2a + self.b2 
    self.intermediate["b2a"] = self.b2a
    return self.b2a



