import ember 

X = ember.Tensor.uniform([100, 5], 0, 20) 
params = ember.Tensor.arange(1, 6, 1).reshape([5, 1])
Y = X @ params 

print(X)

model = ember.MultiLayerPerceptron(10, 10) 

X = ember.Tensor.arange(0, 10, 1).reshape([10, 1])

y = model.forward(X) 

y.backprop(False) 

print(model.W1.grad)
