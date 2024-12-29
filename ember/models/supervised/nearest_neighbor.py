from ember import Tensor
from ember.models.supervised import Regression, Classification
from ember.models import track_access
from typing import Optional, Dict
from ember.datasets import Dataset

class KNearestRegressor(Regression): 
  """Simple K Nearest Neighbors regressor."""

  def __init__(self, dataset: Dataset, K: int, weight: Optional[Tensor] = None): 
    super().__init__() 
    if K <= 0: 
      raise Exception("K must be positive.") 
    if weight is None: 
      # set default weights to be just average
      self.weight = Tensor.ones([K], bidx = 0, requires_grad=False) 
    elif weight.shape != [K]: 
      raise Exception("Input weight should be of shape [K].") 
    else: 
      self.weight = weight
    self._dataset = dataset
    self.K = K
    self.set_parameters() 

  def k_nearest_neighbors(self, X: Tensor) -> Dict[float, Tensor]: 
    # store dict of dist : y_truth. 
    d = dict()
    for x, y in self._dataset: 
      distance = ((x.reshape(self._dataset.D) - X) ** 2).sum().item()
      d[distance] = y 

    return d

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    self.d = self.k_nearest_neighbors(X)
    res = Tensor.zeros(shape=self._dataset[0][1].shape, bidx=0, requires_grad=False) 
    for k in sorted(self.d)[:self.K]: 
      res = res + self.d[k]    # TODO: this is a memory leak, add support for __iadd__ 
    self.z = res * (1/self.K)
    super().forward(X)
    return self.z

class ApproximateKNearestRegressor(Regression): 
  """
  Approximate KNearest Neighbor regressor for datasets with large number of 
  samples or high dimensions. Sacrifices accuracy for inference speed. 
  Essentially takes a random subsample of the data and finds K nearest 
  neighbors.
  """

class KNearestClassifier(Classification): 

  def __init__(self):
    super().__init__() 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()



