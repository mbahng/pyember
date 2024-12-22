import ember 

ds = ember.datasets.LinearDataset(N=20, D=4)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(5) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(1000): 
  loss = None
  for x, y in dl: 
    x.bidx = 1 
    y.bidx = 1
    x.has_grad = False
    y.has_grad = False
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop(False)

    dW = model.W.grad.batchsum().reshape([5], inplace=True) 
    db = model.b.grad.batchsum().reshape([1], inplace=True)

    model.W -= 1e-5 * dW
    model.b -= 1e-5 * db

  if epoch % 25 == 0: 
    print(loss)

