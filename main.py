import ember 

x = ember.Tensor.arange(0, 4, 1) 
y = 2

z = x * y

z.backprop() 

print(z)
print(x.grad) 
