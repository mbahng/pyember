import ember
import networkx as nx
import matplotlib.pyplot as plt 
from ember import Tensor
from ember.models.net import Net 

def to_str(data: list, grad: ember.GradTensor) -> str: 
    data_str = str(data)[2:-2]
    data_str = "\n".join(data_str.split("], ["))
    grad = grad.data
    grad_str = str(grad)[2:-2]
    grad_str = "\n".join(grad_str.split("], ["))
    return f"val={data}\ngrad={grad_str}"


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
        a6.backprop()
        dFdB2 = a6.grad.matmul(a5.grad).matmul(self.b2.grad)

        # update parameters: W1
        return a6, dFdB2

myNet = CustomNet()
X = Tensor([[-1], [2]]) 

for _ in range(100): 
    res, dFdB2 = myNet.forward(X)
    print(res)
    myNet.b2 += dFdB2
