from .. import Tensor, ScalarTensor
import warnings 

class Loss(): 
    def __init__(self): 
        self.loss = None

    def __call__(self, y_truth: Tensor, y_pred: Tensor) -> Tensor:  
      return Tensor([0.0]) 

    def check_has_grad(self, y: Tensor, name="tensor") -> bool: 
      return True if y.has_grad else False

class MSELoss(Loss): 

    def __init__(self): 
      super().__init__() 

    def __call__(self, y_truth: Tensor, y_pred: Tensor) -> Tensor:  
      if y_truth.shape != y_pred.shape: 
        raise Exception(f"The truth shape {y_truth.shape} and predicted shape {y_pred.shape} are not the same. ")
      if self.check_has_grad(y_truth):
        warnings.warn(f"y_truth does has gradients, which will likely backprop the data. Did you mean to do this?") 
      if not self.check_has_grad(y_pred):
        warnings.warn(f"y_pred does not have gradients. You won't be able to backprop the model parameters. ")  
      # N_inv = Tensor([1/(y_truth.shape[0])], shape=[1, 1], has_grad=False) 
      self.y_truth = y_truth
      self.y_pred = y_pred 
      self.diff = self.y_truth - self.y_pred 
      self.sq_diff = self.diff ** 2 
      self.sum_sq_diff = self.sq_diff.sum()  
      self.loss = self.sum_sq_diff 
      return self.loss
