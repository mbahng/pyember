import ember 

ds = ember.datasets.LinearDataset(N=20, D=4)
model = ember.models.LinearRegression(5) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(1000):
  loss = ember.ScalarTensor(0) 
  for j in range(20): 
    x, y = ds.X[j], ds.Y[j].reshape([1, 1], inplace=False, has_grad=False)
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop(False)  
    model.W = model.W - a * model.W.grad.reshape(model.W.shape, inplace=False)
    model.b = model.b - a * model.b.grad.reshape(model.b.shape, inplace=False)

  if epoch % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")

