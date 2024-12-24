import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.MultiLayerPerceptron(15, 10) 
mse = ember.objectives.MSELoss()

for epoch in range(1):  
  loss = None
  for x, y in dl: 
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop() 
    model.step(1e-5)

  print(loss)
