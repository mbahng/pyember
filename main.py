import ember 

A = ember.Tensor([[0, 1], [2, 3]]) 
B = ember.Tensor([[4, 5], [6, 7]]) 
C = A @ B 

C.backprop(True)

print(C)

print(A.grad)
