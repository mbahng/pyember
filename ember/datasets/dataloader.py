from .dataset import Dataset 
import random

class Dataloader(): 

  def __init__(self, dataset: Dataset, batch_size = 1, shuffle = True): 
    self.batch_size = batch_size 
    self.n_batches = len(dataset) // batch_size
    assert(batch_size <= len(dataset))
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
    if self.idx >= self.n_batches: 
      self.idx = 0
      raise StopIteration 

    start = self.batch_size * self.idx 
    end = self.batch_size * (self.idx + 1)

    batch_X = self.dataset.X[start:end]
    batch_Y = self.dataset.Y[start:end] 

    # set batch indices
    batch_X.bidx = 1
    batch_Y.bidx = 1

    self.idx += 1 

    return batch_X, batch_Y


