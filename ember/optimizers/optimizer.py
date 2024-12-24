from abc import ABC, abstractmethod
from ember.models import Model

class Optimizer(ABC): 
   
  def __init__(self, model: Model, lr: float = 1e-4): 
    self.model = model
    self.lr = lr 

  @abstractmethod
  def step(self) -> None:  
    """Abstract method that needs to be implemented for each optimizer type."""

class IterativeOptimizer(Optimizer): 
   
  def __init__(self, model: Model, lr: float): 
    super().__init__(model, lr)

class ClosedOptimizer(Optimizer): 
   
  def __init__(self, model: Model, lr: float): 
    super().__init__(model, lr)

class SGDOptimizer(IterativeOptimizer): 
   
  def __init__(self, model: Model, lr: float): 
    super().__init__(model, lr)

  def step(self) -> None: 
    for n in self.model.parameters():  
      self.model._parameters[n] -= self.lr * self.model._parameters[n].grad.batchsum().reshape(self.model._parameters[n].shape)


