import ember
from ember import Tensor

class CustomNet(): 
    
    def __init__(self): 
        self.a = Tensor([2]) 
        self.b = Tensor([4]) 
        self.c = Tensor([3]) 
        self.d = Tensor([1]) 

    def forward(self, x:ember.Tensor, intermediate) -> ember.Tensor: 

        y1 = self.a * x 
        y = y1 + self.b
        z1 = self.c * y 
        z = z1 + self.d

        top_sort = z.backprop(intermediate)

        # retrieve gradients using chain rule   
        # dFdB2 = a6.grad.matmul(a5.grad).matmul(self.b2.grad)

        # update parameters: W1
        return z, top_sort

myNet = CustomNet()
X = Tensor([10])
res, top_sort = myNet.forward(X, True)

# print(len(top_sort))
# print(f"result : {res}") 
# for tensor in top_sort: 
#     print(tensor.data, tensor.grad)


alpha = Tensor([-0.000001])
for _ in range(20): 
    res, top_sort = myNet.forward(X, False)
    print(res)
    for tensor in top_sort: 
        if tensor == myNet.a: 
            myNet.a += alpha * myNet.a.grad
        elif tensor == myNet.b: 
            myNet.b += alpha * myNet.b.grad
        elif tensor == myNet.c: 
            myNet.c += alpha * myNet.c.grad
        elif tensor == myNet.d: 
            myNet.d += alpha * myNet.d.grad
    print("***********")

