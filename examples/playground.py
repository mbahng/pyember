from ember import Tensor, GradTensor

x = Tensor.gaussian([2, 3]) 

print(x)
print(x.transpose())
