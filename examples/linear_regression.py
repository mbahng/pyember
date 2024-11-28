import ember 

X = ember.Tensor.uniform([20, 5], 0, 10, has_grad=False) 
params = ember.Tensor.arange(1, 6, 1, has_grad=False).reshape([5, 1], inplace=True)
Y_truth = X @ params 
noise = ember.Tensor.gaussian([20, 1], 0, 1, has_grad=False)
Y = Y_truth + noise

model = ember.LinearRegression(5) 
a = 1e-5

for i in range(1000): 
  y = model.forward(X) 
  diff = Y - y 
  squared_diff = diff ** 2 
  loss = squared_diff.sum() 
  loss.backprop(False) 

  model.W = model.W - a * model.W.grad.reshape([5, 1], inplace=False)

  if i % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")
    print(model.W.reshape([5], inplace=False))

  del y, diff, squared_diff, loss 

