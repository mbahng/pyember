import torch 

A = torch.arange(0, 10).reshape(5, 2) 
B = torch.arange(0, 6).reshape(2, 3)
C = A @ B 

print(C)


