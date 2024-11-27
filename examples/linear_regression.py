import ember 

X = ember.Tensor.arange(0, 10, 1).reshape([1, 10])
model = ember.LinearRegression(10) 
model.W = ember.Tensor.arange(10, 20, 1).reshape([10, 1])

print(X)
print(model.W)
print(model.b)

y = model.forward(X) 
print(y)
print("===============")
y.backprop(False) 

print(X.grad)
print(model.W.grad)
