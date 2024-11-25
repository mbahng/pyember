from ember import Tensor

x = Tensor.arange(0, 6, 1)
y = Tensor.arange(0, 6, 1)

z = x + y 

z.backprop(False)


print(x.grad)
