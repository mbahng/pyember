from ember import Tensor
from abc import ABC, abstractmethod

def track_access(original_method):
  """Decorator for keeping track of variables accessed in order."""
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

class Model(ABC): 

  def __init__(self): 
    self._parameters = {}
    self._intermediate = dict() 
    self._forward_called = False
    self._nonparams = set() 
    self._nonparams = set(vars(self)) 

  def set_parameters(self): 
    for k in vars(self):
      if k not in self._nonparams: 
        self._parameters[k] = vars(self)[k]

  def parameters(self): 
    return self._parameters

  def intermediate(self): 
    if not self._forward_called: 
      raise Exception("Call forward to load intermediate values.")

    return self._intermediate

  @abstractmethod
  def forward(self, X: Tensor) -> Tensor: 
    self._forward_called = True
