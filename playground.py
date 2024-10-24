import ember 
import networkx as nx
import matplotlib.pyplot as plt 
from pprint import pprint

def to_str(data: list, grad: list) -> str: 
    grad_str = str(grad)[2:-2]
    grad_str = "\n".join(grad_str.split("], ["))
    return f"val={data}\ngrad={grad_str}"

def first():
    a = ember.Tensor([2, -3]) 
    h = a ** 2
    b = ember.Tensor([3, 5])

    c = b * h

    d = ember.Tensor([10, 1])
    e = c.dot(d) 

    f = ember.Tensor([-2])

    g = f * e
    top_sort = g.backprop()

    adj_list = {to_str(t.data, t.grad) : [] for t in top_sort}

    for node in top_sort: 
        for t in node.prev: 
            adj_list[to_str(t.data, t.grad)].append(to_str(node.data, node.grad))


    G = nx.DiGraph(adj_list) 
    nx.draw_planar(G, with_labels=True, node_size=8000)
    plt.show() 

def second(): 

    x = ember.Tensor([[1, 2, 3], [4, 5, 6]]) # 2 x 3
    y =  ember.Tensor([[1, 2], [4, 4]]) # 2 x 2
    z = y.matmul(x) 

    top_sort = z.backprop() 

    adj_list = {to_str(t.data, t.grad) : [] for t in top_sort}

    for node in top_sort: 
        for t in node.prev: 
            adj_list[to_str(t.data, t.grad)].append(to_str(node.data, node.grad))


    G = nx.DiGraph(adj_list) 
    nx.draw_planar(G, with_labels=True, node_size=8000)
    plt.show() 

def third(): 

    a = ember.Tensor([[1, 2, 3], [4, 5, 6]]) # 2 x 3
    b =  ember.Tensor([[1, 2], [4, 8]]) # 2 x 2
    c = b.matmul(a)

    top_sort = c.backprop() 

    adj_list = {to_str(t.data, t.grad) : [] for t in top_sort}

    for node in top_sort: 
        for t in node.prev: 
            adj_list[to_str(t.data, t.grad)].append(to_str(node.data, node.grad))

    pprint(a.grad)
    pprint(b.grad)

    G = nx.DiGraph(adj_list) 
    nx.draw_planar(G, with_labels=True, node_size=8000)
    plt.show() 

first()
