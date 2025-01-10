import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
model = ember.models.MultiLayerPerceptron(15, 10) 
mse = ember.objectives.MSELoss()
a = 1e-4

for epoch in range(500):  
  loss = ember.ScalarTensor(0)
  for j in range(20): 
    x, y = ds.X[j], ds.Y[j].reshape([1, 1], inplace=False, has_grad=False)
    y_ = model.forward(x) 
    loss = mse(y, y_)
    loss.backprop(False) 

    model.W1 = model.W1 - a * model.W1.grad.reshape(model.W1.shape, inplace=False)
    model.b1 = model.b1 - a * model.b1.grad.reshape(model.b1.shape, inplace=False)
    model.W2 = model.W2 - a * model.W2.grad.reshape(model.W2.shape, inplace=False)
    model.b2 = model.b2 - a * model.b2.grad.reshape(model.b2.shape, inplace=False)

  if epoch % 25 == 0: 
    print(f"LOSS = {loss[0].item()}")
    # print(model.W1.reshape([5], inplace=False))


