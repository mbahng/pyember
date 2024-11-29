from ember import Tensor

x = Tensor.gaussian([3, 3]) 
y = Tensor.gaussian([3, 3])  

z = x * y 
z.backprop(True)

print(x.grad)
print(y.grad)
print(z.grad)
