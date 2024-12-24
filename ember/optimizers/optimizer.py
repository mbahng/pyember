from abc import ABC
from ember.models import Model 
from ember.objectives import Loss

class Optimizer(ABC): 
   
    def __init__(self, model: Model, loss: Loss): 
        pass 

class IterativeOptimizer(Optimizer): 
   
   def __init__(self, model: Model, loss: Loss, lr: float): 
       self.model = model 
       self.loss = loss
       self.lr = lr

class ClosedOptimizer(Optimizer): 
   
   def __init__(self): 
       pass 

class SGDOptimizer(IterativeOptimizer): 
   
   def __init__(self, model, loss, lr): 
       super().__init__(model, loss, lr)

class AdamOptimizer(IterativeOptimizer): 
   
   def __init__(self, model, loss, lr): 
       super().__init__(model, loss, lr)

class NesterovOptimizer(IterativeOptimizer): 
   
   def __init__(self, model, loss, lr): 
       super().__init__(model, loss, lr)

class NewtonOptimizer(IterativeOptimizer): 
   
   def __init__(self, model, loss, lr): 
       super().__init__(model, loss, lr)

class LBFGSOptimizer(IterativeOptimizer): 
   
   def __init__(self, model, loss, lr): 
       super().__init__(model, loss, lr)


