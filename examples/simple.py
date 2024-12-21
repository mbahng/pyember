from ember import GradTensor, Tensor

def testA(): 
  A = Tensor.gaussian([2], 0, 1, 0, True)
  B = Tensor.gaussian([3, 2], 0, 1, 1, True) 
  C = A + B 


  D = Tensor.gaussian([2], 0, 1, 0, True)
  print(C.nb_indices()) 
  print(D.nb_indices()) 
  E = C * D

  E.backprop(False)

  print(A.grad.shape)
  print(B.grad.shape)
  print(C.grad.shape)
  print(D.grad.shape)
  print(E.grad.shape)

def testB(): 
  A = Tensor.gaussian([4, 2], 0, 1, 1, True)
  B = Tensor.gaussian([3, 2], 0, 1, 1, True) 
  C = A + B 

  D = Tensor.gaussian([2], 0, 1, 0, True)
  E = D * C

  E.backprop(False)


  print(A.grad.shape)
  print(B.grad.shape)
  print(C.grad.shape)
  print(D.grad.shape)
  print(E.grad.shape)


A = Tensor.gaussian([2, 2, 3], 0, 1, 1, True)
B = Tensor.gaussian([4, 3, 2], 0, 1, 1, True) 
C = A @ B 
print(C)

C.backprop(True)
