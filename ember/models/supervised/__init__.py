from ember.models import Model

class Regression(Model): 

  def __init__(self): 
    super().__init__()

  def set_parameters(self): 
    super().set_parameters() 

class Classification(Model): 

  def __init__(self): 
    super().__init__() 

  def set_parameters(self): 
    super().set_parameters() 


__all__ =  [
  "Regression", 
  "Classification"
]
