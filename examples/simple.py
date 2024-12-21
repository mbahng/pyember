from ember import GradTensor, Tensor

data = list(range(6)) 
A = GradTensor(data + data, [2, 2, 3], 2, 1)
B = GradTensor(data, [3, 2], 1, 0) 
print(A) 
print(B)

print(A @ B)
