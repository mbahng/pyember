import ember 

ds = ember.datasets.LinearDataset(N=20, D=15)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(15) 
mse = ember.objectives.MSELoss() 
optim = ember.optimizers.SGDOptimizer(model, 1e-4)

for epoch in range(1000): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop()
    optim.step()
  #
  if epoch % 100 == 0: 
    print("========================================")
    print(loss)
    print(ds.truth_param.reshape([15])) 
    print(model.W) 
    print(model.b)
