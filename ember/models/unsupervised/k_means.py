from ember import Tensor
from ember.models.unsupervised import Clustering
from ember.models import track_access

class KMeans(Clustering): 

  def __init__(self):
    super().__init__() 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()

