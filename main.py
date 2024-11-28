from ember import Tensor 
from ember.datasets import LinearDataset 

ds = LinearDataset(N=100, D=10, noise="gaussian")

print(ds.X)
