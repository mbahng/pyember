from ember import Tensor, GradTensor

x = Tensor.arange(0, 6)
print(x.grad)

