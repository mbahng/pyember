from ember import Tensor
from ember.models.supervised import Regression
from ember.models import track_access

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

class BayesianLinearRegression(Regression): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = Tensor.gaussian([input_dim, 1], 0, 1)
    self.b = Tensor.gaussian([20, 1], 0, 1)

  def forward(self, X: Tensor): 
    self.z1 = X @ self.W
    z = self.z1 + self.b
    return z


