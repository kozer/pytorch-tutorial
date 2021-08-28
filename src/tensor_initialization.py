import torch

# ----------------------------------------- #
# Tensor initialization                     #
# ----------------------------------------- #

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
my_tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
my_tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6]],
                          dtype=torch.float32,
                          device="cuda")

my_tensor4 = torch.tensor([[1, 2, 3], [4, 5, 6]],
                          dtype=torch.float32,
                          device="cpu")

my_tensor5 = torch.tensor([[1, 2, 3], [4, 5, 6]],
                          dtype=torch.float32,
                          device="cuda", requires_grad=True)

print(my_tensor)
print(my_tensor2)
print(my_tensor3)
print(my_tensor4)
print(my_tensor5)

# Other initialization methods

x = torch.empty(size=(3,3))
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
print(x)
x = torch.ones((3,3))
print(x)
x = torch.eye(5,5)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1,5)).uniform_(0,1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# How to initialise and convert tensors to other types (int, float, double)
x = torch.arange(4)
print(x)
print(x.bool())
print(x.short())
print(x.long())
print(x.half())
print(x.float())
print(x.double())

#Array to tensor conversion
import numpy as np

np_arr = np.zeros((5,5))
tensor = torch.from_numpy(np_arr)
print(x)
print(tensor.numpy())

