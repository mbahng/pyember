import ember 
D = 10 
X = ember.Tensor.uniform([100, 10], 0, 1)

model = ember.LinearRegression(D)

for i in range(100): 
    x = X[i]
    y = model.forward(x) 
    print(y.data()[0])
    g = model.W.grad.copy().reshape(model.W.shape)
    model.W += g 

