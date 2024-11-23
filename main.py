import torch

x = torch.randn(2, 2, requires_grad=True)
y = torch.randn(2, 2, requires_grad=True)
print(x.grad)

z = x + y
loss = z.sum()

loss.backward()
print(x.grad)

