import ember 
a = ember.Tensor([2, -3]) 
h = a ** 2
b = ember.Tensor([3, 5])

c = b * h

d = ember.Tensor([10, 1])
e = c.dot(d) 

f = ember.Tensor([-2])

g = f + e
top_sort = g.backprop()

adj_list = {to_str(t.data, t.grad) : [] for t in top_sort}

for node in top_sort: 
  for t in node.prev: 
    adj_list[to_str(t.data, t.grad)].append(to_str(node.data, node.grad))
