import ember 

x = ember.Tensor.gaussian([4, 1], 0, 1)
y = ember.Tensor.ones([4, 4])
z = y @ x 
print(x) 
print(y) 
print(z)

z.backprop(False) 

print(x.grad)


