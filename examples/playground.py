from ember import Tensor 

a = Tensor([2, -3]) # [2, -3]
h = a ** 2          # [4, 9]
b = Tensor([3, 5])  # [3, 5]

c = b * h           # [12, 45]

d = Tensor([10, 1]) # [10, 1]
e = c.dot(d)        # 165

f = Tensor(-2)      # -2

g = f * e           # -330

top_sort = g.backprop()
print(a.grad) # [-240, +60]
print(h.grad) # [-60, -10]
print(b.grad) # [-80, -18]
print(c.grad) # [-20, -2]
print(d.grad) # [-24, -90]
print(e.grad) # [-2]
print(f.grad) # [165]
print(g.grad) # [1]
