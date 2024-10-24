from ember import Tensor, GradTensor 


x = Tensor([1, 2])
y = x.sum()

y.backprop()

x_grad = x.grad 

print(x + x_grad)
