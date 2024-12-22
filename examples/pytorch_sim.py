import torch 

A = torch.ones(2, 3, 2, 3) 
B = torch.randn(2, 3) 
C = A + B 
D = C.sum() 
print(D)
D.backward()

# (1)         (1)
# (1, 1)      (1, 1)
# (1) 
# (1, 2, 1)   (1, 2, 1)
# (2, 1)      
# (2, 1, 1)   (1, 1)
# (2, 1) 
# (2, 1, 1)   (2, 1, 1)
