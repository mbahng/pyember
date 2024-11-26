import ember
from ember import Tensor    

class CustomNet(): 
  
  def __init__(self): 
    self.a = Tensor([2]) 
    self.b = Tensor([4]) 
    self.c = Tensor([3]) 
    self.d = Tensor([1]) 

  def forward(self, x:ember.Tensor, intermediate) -> ember.Tensor: 

    y1 = self.a * x 
    y = y1 + self.b
    z1 = self.c * y 
    z = z1 + self.d

    top_sort = z.backprop(intermediate)
    # top_sort = []

    # update parameters: W1
    return z, top_sort

myNet = CustomNet()
X = Tensor([10])
res, top_sort = myNet.forward(X, True)

# print(len(top_sort))
# print(f"result : {res}") 
# for tensor in top_sort: 
#     print(tensor.data, tensor.grad)


alpha = Tensor([-0.001])
for i in range(150): 
  print(i)
  print(myNet.a.grad.shape)
  print(myNet.b.grad.shape)
  print(myNet.c.grad.shape)
  print(myNet.d.grad.shape)

  res, top_sort = myNet.forward(X, False)
  print(res)
  for tensor in top_sort: 
    if tensor == myNet.a: 
      myNet.a += alpha * myNet.a.grad
    elif tensor == myNet.b: 
      myNet.b += alpha * myNet.b.grad
    elif tensor == myNet.c: 
      myNet.c += alpha * myNet.c.grad
    elif tensor == myNet.d: 
      myNet.d += alpha * myNet.d.grad

  print("***********")

