from ember import Tensor, ScalarTensor
from typing import Union

def track_access(original_method):
  def wrapper(self, X):
    self._intermediate = {"X" : X}
    
    original_getattribute = self.__class__.__getattribute__
    
    def __getattribute__(self, name):
        value = original_getattribute(self, name)
        if isinstance(value, Tensor):  # Only track Tensor objects
            self._intermediate[name] = value
        return value
        
    self.__class__.__getattribute__ = __getattribute__
    
    result = original_method(self, X)
    
    self.__class__.__getattribute__ = original_getattribute
        
    return result
  return wrapper

class Model(): 

  def __init__(self): 
    self._intermediate = dict() 
    self._forward_called = False
    self._nonparams = set()
    self._nonparams = set(vars(self)) 

  def set_parameters(self): 
    _parameters = dict()
    for k, v in vars(self).items(): 
      if k not in self._nonparams: 
        _parameters[k] = v 
    self._parameters = _parameters 
    self._nonparams.add("_parameters")

  def parameters(self): 
    return self._parameters

  def intermediate(self): 
    if not self.forward_called: 
      raise Exception("Call forward to load intermediate values.")

    return self._intermediate

  def forward(self, X: Tensor) -> Tensor: 
    self.forward_called = True

  def step(self, a: Union[ScalarTensor, float, int]) -> None: 
    pass

