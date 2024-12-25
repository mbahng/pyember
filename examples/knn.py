import ember
from ember.models import KNearestRegressor
from ember.datasets import LinearDataset

ds = LinearDataset(N=20, D=3)
model = KNearestRegressor(dataset=ds, K=1)
mse = ember.objectives.MSELoss() 

# hyperparameter tuning
for k in range(1, 21):
  model.K = k
  print(f"{k} ===") 
  loss = 0
  for i in range(len(ds)): 
    x, y = ds[i] 
    y_ = model.forward(x) 
    loss = loss + mse(y, y_) 

  print(loss)

