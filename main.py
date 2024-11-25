from ember import Tensor

x = Tensor.uniform([4], 0, 1)
y = Tensor.uniform([4], 0, 1)
w = Tensor.uniform([4], 0, 1) 

z = x * y 

a = z + w

a.backprop(True)

print(x.grad)
