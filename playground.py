import ember 
import networkx as nx
import matplotlib.pyplot as plt 

def first():
    a = ember.Tensor([2, -3])
    b = ember.Tensor([3, 5])

    c = a * b

    d = ember.Tensor([10, 1])
    e = c.dot(d) 

    f = ember.Tensor([-2])

    g = f * e
    top_sort = g.backprop()

    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)
    print(e.grad)
    print(f.grad)
    print(g.grad)

    def to_str(data: list, grad: list) -> str: 
        return f"val={data}\ngrad={grad}"
        

    adj_list = {to_str(t.data, t.grad) : [] for t in top_sort}

    for node in top_sort: 
        for t in node.prev: 
            adj_list[to_str(t.data, t.grad)].append(to_str(node.data, node.grad))


    G = nx.DiGraph(adj_list) 
    nx.draw_planar(G, with_labels=True, node_size=8000)
    plt.show() 

first()
