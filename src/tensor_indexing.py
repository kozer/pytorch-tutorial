import torch

# ----------------------------------------- #
# Tensor indexing                           #
# ----------------------------------------- #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# Get all features first batch
print(x[0])  # x[0,:]
print(x[0].shape)  # x[0,:]
# Get 1 feature all batches
print(x[:, 0])
print(x[:, 0].shape)
#3d row 0 - 10
print(x[2, 0:10])  # 0:10 -> [1,2,3,4...9]
print(x[2, 0:10].shape)

#Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
print('all elems', x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])
print(x[rows, cols].shape)

# More advance indexing
x = torch.arange(10)
print('all elems', x)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x * 2))
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
print(x.ndimension())  # 5x5x5 ---> 3
print(x.numel())
