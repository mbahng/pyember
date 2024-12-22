import ember
from ember import GradTensor

A = GradTensor([1, 2, 3, 4] * 2, [2, 2, 2], 2) 
B = GradTensor([1, 2, 3, 4], [2, 2], 1) 

C = A @ B 

print(C)
