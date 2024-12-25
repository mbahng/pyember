from ember import Tensor
from ember.models.supervised import Regression
from ember.models import track_access

class MultiLayerPerceptron(Regression): 

  def __init__(self, input_dim: int, hidden_dim:int): 
    super().__init__() 
    self.W1 = Tensor.gaussian([input_dim, hidden_dim], 0, 1) 
    self.b1 = Tensor.gaussian([hidden_dim], 0, 1) 
    self.W2 = Tensor.gaussian([hidden_dim], 0, 1)
    self.b2 = Tensor.gaussian([1], 0, 1) 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor): 
    self.W1a = X @ self.W1 
    self.b1a = self.W1a + self.b1
    self.o1a = self.b1a.relu() 
    self.W2a = self.o1a.dot(self.W2) 
    self.b2a = self.W2a + self.b2 

    super().forward(X)
    return self.b2a



