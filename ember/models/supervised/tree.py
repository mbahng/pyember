from ember import Tensor
from ember.models.supervised import Regression, Classification
from ember.models import track_access

class RegressionTree(Regression): 

  def __init__(self):
    super().__init__() 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()

class ClassificationTree(Classification): 

  def __init__(self):
    super().__init__() 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()

