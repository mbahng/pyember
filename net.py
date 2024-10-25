import ember
import networkx as nx
import matplotlib.pyplot as plt 
from ember import Tensor
from ember.models.net import Net 

class CustomNet(Net): 
    
    def __init__(self): 
        self.W1 = Tensor.gaussian([2, 2]) 
        self.b1 = Tensor.gaussian([2, 1])

        self.W2 = Tensor.gaussian([2, 2]) 
        self.b2 = Tensor.gaussian([2, 1])

    def forward(self, X:ember.Tensor) -> ember.Tensor: 

        a1 = self.W1.matmul(X)
        a2 = a1 + self.b1 
        
        a3 = a2.relu() 
        a4 = self.W2.matmul(a3) 
        a5 = a4 + self.b2 
        
        a6 = a5.sum()

        # backpropagate gradients 
        top_sort = a6.backprop()

        # retrieve gradients using chain rule   
        # dFdB2 = a6.grad.matmul(a5.grad).matmul(self.b2.grad)

        # update parameters: W1
        return a6, top_sort

myNet = CustomNet()
X = Tensor.gaussian([2, 1])
res, top_sort = myNet.forward(X)

print(f"result : {res}") 

for tensor in top_sort: 
    print(tensor.grad)

