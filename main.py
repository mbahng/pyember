from ember import Tensor, GradTensor

x = Tensor.arange(0, 6, 1)
sq = x ** 2 

print(x.grad)

