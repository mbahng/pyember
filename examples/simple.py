import ember 

ds = ember.datasets.LinearDataset(N=20, D=14) 
dl = ember.datasets.Dataloader(ds, batch_size=2) 

for batch in dl: 
  print(batch)
  break

