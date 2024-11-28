import ember
from .model import Model

class Regression(Model): 

  def __init__(self): 
    super().__init__()
    self.params = {}
