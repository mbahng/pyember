from os import walk
from ember import Tensor, GradTensor

class LinearRegression(): 

  def __init__(self, input_dim: int):
    super().__init__() 
    self.W = Tensor.gaussian([input_dim, 1], 0, 1)
    self.b = Tensor.gaussian([1, 1], 0, 1)

  def forward(self, X: Tensor): 
    z1 = X @ self.W 
    return z1


model = LinearRegression(10)


X = Tensor.gaussian([1, 10], 0, 1) 
y = model.forward(X)

y.backprop(True) 
print(y)

print(model.W)
print(model.W.grad)


