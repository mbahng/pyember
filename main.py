from ember import Tensor

x = Tensor.arange(0, 4, 1).reshape([2, 2])
y = Tensor.arange(4, 8, 1).reshape([2, 2])

z = x @ y 

print(z)

z.backprop(True)

print(x.grad[0,0,:,:].pivot)
print(x.grad[0,1,:,:])
print(x.grad[1,0,:,:])
print(x.grad[1,1,:,:])
