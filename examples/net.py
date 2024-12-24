import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
dl = ember.datasets.Dataloader(ds, batch_size=2)
model = ember.models.MultiLayerPerceptron(15, 10) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(500):  
  loss = None
  for x, y in dl: 
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop() 

    # print(f"X --- {x.meta()}") 
    #
    # for name, tensor in model.intermediate.items(): 
    #   print(f"{name} --- {tensor.grad.meta()}") 
    # for name, tensor in mse.intermediate.items(): 
    #   print(f"{name} --- {tensor.grad.meta()}") 

    dW1 = model.W1.grad.batchsum().reshape(model.W1.shape, inplace=False) 
    db1 = model.b1.grad.batchsum().reshape(model.b1.shape, inplace=False)
    dW2 = model.W2.grad.batchsum().reshape(model.W2.shape, inplace=False) 
    db2 = model.b2.grad.batchsum().reshape(model.b2.shape, inplace=False)

    model.W1 -= 1e-5 * dW1
    model.b1 -= 1e-5 * db1
    model.W2 -= 1e-5 * dW2
    model.b2 -= 1e-5 * db2
   
  if epoch % 25 == 0: 
    print(f"LOSS = {loss}")


