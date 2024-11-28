import ember 

X = ember.Tensor.uniform([20, 5], 0, 10) 
# need to create a copy since gradient arrives at X twice in computation graph 
X_copy = X.copy() 
params = ember.Tensor.arange(1, 6, 1).reshape([5, 1])
Y = X_copy @ params 
# noise = ember.Tensor.gaussian([20, 1], 0, 1) 
# Y = Y_truth + noise

model = ember.LinearRegression(5) 
a = 1e-5

for i in range(1000): 
  y = model.forward(X) 
  diff = Y - y 
  squared_diff = diff ** 2 
  loss = squared_diff.sum() 
  loss.backprop(False) 

  for j in range(5): 
    model.W[j, 0] = model.W[j,0].item() - a * model.W.grad[0,j,0].item()

  if i % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")
    print(model.W.copy().reshape([5]))

  del y, diff, squared_diff, loss 

