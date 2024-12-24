import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(15) 
mse = ember.objectives.MSELoss() 
optim = ember.optimizers.SGDOptimizer(model, 1e-5)

for epoch in range(500): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop()
    optim.step()

  if epoch % 25 == 0: 
    print(loss)
