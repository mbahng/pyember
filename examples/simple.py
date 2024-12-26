from ember.datasets import LinearDataset 

ds = LinearDataset(10, 3) 
feature_idx = 1
dataset = sorted(
  ds.items(), 
  key = lambda x : x[0][feature_idx].item()
)

print(dataset)

y_upper = y_lower = None 
x_threshold = None  
lower_ys = deque([])
upper_ys = deque([row[1].item() for row in dataset])

for i in range(len(dataset) - 1): 
  # calculate the x_threshold 
  x_threshold = (dataset[i][0][feature_idx] + dataset[i + 1][0][feature_idx]).item() / 2 
  lower_ys.append(upper_ys.popleft())
  y_lower = sum(lower_ys) / len(lower_ys)
  y_upper = sum(upper_ys) / len(upper_ys)
  se_lower = sum([(ly - y_lower) ** 2 for ly in lower_ys])
  se_upper = sum([(uy - y_upper) ** 2 for uy in upper_ys])
  mse = (se_lower + se_upper) / len(dataset)

  print(mse)
