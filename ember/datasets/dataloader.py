from .. import Tensor 
from .dataset import Dataset 
import random

class Dataloader(): 

  def __init__(self, dataset: Dataset, batch_size = 1, shuffle = True): 
    self.n_batches = (len(dataset) % batch_size)
    self.batch_size = batch_size 
    assert(batch_size < len(dataset) and  self.n_batches == 0)
    self.dataset = dataset 
    self.shuffle = shuffle
    self.idx = 0 
    self.order = self.get_order()

  def get_order(self): 
    order = list(range(len(self.dataset))) 
    if self.shuffle: 
      order = random.shuffle(order)
    return order

  def __len__(self): 
    return self.n_batches  

  def __iter__(self): 
    return self 

  def __next__(self): 
    if self.idx > len(self.dataset): 
      raise StopIteration 

    batch_X = self.dataset.X[self.idx:self.idx + self.batch_size]
    batch_Y = self.dataset.Y[self.idx:self.idx + self.batch_size]
    self.idx += 1 

    return batch_X, batch_Y


