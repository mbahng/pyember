import ember 

ds = ember.datasets.LinearDataset(N=20, D=4)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(5) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(1000):
  loss = ember.ScalarTensor(0) 
  for x, y in dl: 
    y_ = model.forward(x) 
    y_.backprop(True)
    loss = mse(y, y_)
    print(loss)
    exit()
    model.W = model.W - a * model.W.grad.reshape(model.W.shape, inplace=False)
    model.b = model.b - a * model.b.grad.reshape(model.b.shape, inplace=False)

  if epoch % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")

