import torch

# ----------------------------------------- #
# Tensor math and comparison operations     #
# ----------------------------------------- #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)
print(torch.add(x, y))
print(x + y)

#Subtraction
print(x - y)

#Division
print(torch.true_divide(x, y))

# Inplace operations
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x
print(t)

# Exponentiation
z = x.pow(2)
z = x**x
print(z)

# Simple comparison
z = x > 0
print(z)
z = x < 0
print(z)

# Matrix manipulation
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
print(torch.mm(x1, x2))  # 2x3
print(x1.mm(x2))  # 2x3

# Matrix exponentiation
m = torch.rand(5, 5)
print(m.matrix_power(3))
print(m.matrix_exp())

#Element wise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix mult
batch = 32
n = 10
m = 20
p = 30
t1 = torch.rand(batch, n, m)
t2 = torch.rand(batch, m, p)
out_bmm = torch.bmm(t1, t2)
print(out_bmm.size())

#Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
print(z)
z = x1**x2
print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
print(sum_x)
print(torch.max(x, dim=0))
print(torch.min(x, dim=0))
print(torch.abs(x))
#same as max, returns index
print(torch.argmax(x, dim=0))
#same as min, returns index
print(torch.argmin(x, dim=0))
print(torch.mean(x.float(), dim=0))
print(torch.eq(x, y))
print(torch.sort(y, dim=0, descending=False))

z = torch.clamp(x, min=0, max=10)
print(z)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)
