import ember 

ds = ember.datasets.LinearDataset(N=20, D=14)
model = ember.models.RegressionTree(ds) 

print(len(model.leaves))
model.step()
print(len(model.leaves))
print(model.root)
print(model.root.left)
print(model.root.right)
model.step()
print(len(model.leaves))
