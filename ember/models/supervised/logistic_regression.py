from ember import Tensor
from ember.models.supervised import Regression
from ember.models import track_access


class LogisticRegression(Regression): 
  """Simple logistic regression for binary classification."""

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


class SoftmaxRegression(Regression): 
  """Simple softmax regression for multi-class classification.""" 

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


