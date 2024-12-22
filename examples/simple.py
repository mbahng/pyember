import ember

ds = ember.datasets.LinearDataset(N=20, D=14) 
dl = ember.datasets.Dataloader(ds, batch_size=2) 


A = ember.Tensor.arange(0, 12, 1).reshape([2, 6]) 
A.bidx = 1
B = ember.Tensor.arange(0, 6, 1)
C = A.dot(B) 

print(C)
C.backprop(False) 

print(A.grad) 
print(B.grad) 
print(C.grad)
