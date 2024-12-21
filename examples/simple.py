from ember import GradTensor, Tensor

A = Tensor.gaussian([2, 3, 2], 0, 1, 1, True)
B = Tensor.gaussian([4, 3, 2], 0, 1, 1, True) 
C = A + B
print(C) 
C.backprop(False) 

print(A.grad.shape, A.grad.bidx, A.grad.pidx)
print(B.grad.shape, B.grad.bidx, B.grad.pidx)
print(C.grad.shape, C.grad.bidx, C.grad.pidx)
