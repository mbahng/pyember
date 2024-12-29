from .. import Tensor 
from abc import ABC, abstractmethod
import warnings 

class Loss(ABC): 
  def __init__(self): 
    self.loss = None 
    self.intermediate = dict()

  @abstractmethod
  def __call__(self, y_truth: Tensor, y_pred: Tensor) -> Tensor:  
    '''Abstract method that must be implemented by subclasses'''
    pass

class MSELoss(Loss): 

    def __init__(self): 
      super().__init__() 

    def __call__(self, y_truth: Tensor, y_pred: Tensor) -> Tensor:  
      if y_truth.shape != y_pred.shape: 
        raise Exception(f"The truth shape {y_truth.shape} and predicted shape {y_pred.shape} are not the same. ")
      if y_truth.requires_grad:
        warnings.warn(f"y_truth does has gradients, which will likely backprop the data. Did you mean to do this?") 
      if not y_pred.requires_grad:
        warnings.warn(f"y_pred does not have gradients. You won't be able to backprop the model parameters. ")  
      self.y_truth = y_truth
      self.y_pred = y_pred 

      self.diff = self.y_truth - self.y_pred 
      self.intermediate["diff"] = self.diff

      self.sq_diff = self.diff ** 2  
      self.intermediate["sq_diff"] = self.sq_diff

      self.sum_sq_diff = self.sq_diff.sum()  
      self.intermediate["sum_sq_diff"] = self.sum_sq_diff

      self.loss = self.sum_sq_diff 
      self.intermediate["loss"] = self.sum_sq_diff

      return self.loss
