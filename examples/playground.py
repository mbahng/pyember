from ember import Tensor, GradTensor

x = GradTensor([1, 2, 3, 4, 5, 6], [2, 3], 0, 1) 

print(x.transpose()) 
