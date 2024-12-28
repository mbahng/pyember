import ember 
import networkx as nx
import matplotlib.pyplot as plt 

def rep(t: ember.Tensor): 
  return f"{t.shape}\n{t.grad.shape}"

a = ember.Tensor([[1, 2, 3, 4], [4, 5, 6, 7]]) # 2 x 4
b =  ember.Tensor([[0, 1], [1, 2], [4, 8]]) # 3 x 2
c = b.matmul(a)

top_sort = c.backprop() 

print(top_sort)

adj_list = {rep(t) : [] for t in top_sort}

for node in top_sort: 
    for t in node.prev: 
        adj_list[rep(t)].append(rep(node))

print(a.grad)
print(b.grad)

G = nx.DiGraph(adj_list)  # type: ignore
nx.draw_planar(G, with_labels=True, node_size=8000)
plt.show() 

