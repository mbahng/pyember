from ember import Tensor 

x = Tensor(2.13) 
y = x.copy() 
print(id(x.storage))
print(id(y.storage))
