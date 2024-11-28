import ember 

X = ember.Tensor.uniform([20, 15], 0, 10, has_grad=False) 
params = ember.Tensor.arange(1, 16, 1, has_grad=False).reshape([15, 1], inplace=True)
Y_truth = X @ params 
noise = ember.Tensor.gaussian([20, 1], 0, 1, has_grad=False)
Y = Y_truth + noise

model = ember.MultiLayerPerceptron(15, 10) 
a = 1e-4

for epoch in range(500):  
  loss = ember.ScalarTensor(0)
  for i in range(20): 
    x = X[i] # (1, 15)
    y = Y[i] # (1, 1)
    y_ = model.forward(x) 
    diff = y - y_
    squared_diff = diff ** 2 
    loss = squared_diff.sum() 
    loss.backprop(False) 

    model.W1 = model.W1 - a * model.W1.grad.reshape(model.W1.shape, inplace=False)
    model.b1 = model.b1 - a * model.b1.grad.reshape(model.b1.shape, inplace=False)
    model.W2 = model.W2 - a * model.W2.grad.reshape(model.W2.shape, inplace=False)
    model.b2 = model.b2 - a * model.b2.grad.reshape(model.b2.shape, inplace=False)

  if epoch % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")
    # print(model.W1.reshape([5], inplace=False))


