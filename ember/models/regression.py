import ember
from . import Model
from .. import Tensor

class Regression(Model): 

    def __init__(self): 
        super().__init__()
        self.params = {}

class LinearRegression(Regression): 

    def __init__(self, input_dim: int):
        super().__init__() 
        self.W = ember.Tensor.gaussian([input_dim, 1], 0, 1)
        self.b = ember.Tensor.gaussian([1, 1], 0, 1)

    def forward(self, X: Tensor): 
        z1 = X @ self.W 
        z = z1 + self.b
        # backprop must be in here, can't be called globally for some reason
        z.backprop(True) # False doesn't update gradients for some reason
        return z
