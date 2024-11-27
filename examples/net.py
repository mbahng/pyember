import ember 

model = ember.MultiLayerPerceptron(10, 10) 

X = ember.Tensor.arange(0, 10, 1).reshape([10, 1])

y = model.forward(X) 

print(y)

y.backprop(False) 

print(model.W1.grad)
