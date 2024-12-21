import torch 

x = torch.randn(4, 3, 2) 
y = torch.randn(7, 3, 2)
print((x + y).shape)
