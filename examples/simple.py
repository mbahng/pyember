import ember 

# generate dataset with 100 samples, 8 features
X = ember.Tensor.arange(0, 3, 1).reshape([3, 1])
W = ember.Tensor.arange(3, 6, 1).reshape([1, 3])
b = ember.Tensor([[0]])
z1 = W @ X 

print(X.grad)
print(W.grad)
print(b.grad)
print(z1.grad)
