from ember import Tensor, GradTensor

x = Tensor.arange(0, 6, 1, has_grad=False)
y = Tensor.arange(0, 6, 1, has_grad=True)
z = x * y 
print(x.has_grad)
print(y.has_grad)

print(z) 
z.backprop(False)

print(y.grad)
