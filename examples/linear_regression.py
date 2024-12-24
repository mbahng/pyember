import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.LinearRegression(15) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(1000): 
  loss = None
  for x, y in dl: 
    y_ = model.forward(x)  
    loss = mse(y, y_)
    loss.backprop()

    print(f"X --- {x.meta()}") 

    for name, tensor in model.intermediate.items(): 
      print(f"{name} --- {tensor.grad.meta()}") 
    for name, tensor in mse.intermediate.items(): 
      print(f"{name} --- {tensor.grad.meta()}") 

    dW = model.W.grad.batchsum().reshape([5], inplace=True) 
    db = model.b.grad.batchsum().reshape([1], inplace=True)
   
    model.W -= 1e-5 * dW
    model.b -= 1e-5 * db
  
  if epoch % 25 == 0: 
    print(loss)
  
