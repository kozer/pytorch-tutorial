import torch

# ----------------------------------------- #
# Tensor reshape                            #
# ----------------------------------------- #

x = torch.arange(9)

x_3x3 = x.view(3, 3)
print(x_3x3)

x_3x3 = x.reshape(3, 3)
print(x_3x3)

y = x_3x3.t()
print(y)
# print(y.view(9)) - error
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
print(x1)
x2 = torch.rand((2, 5))
print(x2)
print(torch.cat((x1, x2), dim=0))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1))
print(torch.cat((x1, x2), dim=1).shape)

# unroll, flatten
print(x1.view(-1))
print(x1.view(-1).shape)

batch = 64
x = torch.rand((batch, 2, 5))

print(x.view(batch, -1))
print(x.view(batch, -1).shape)
print(x.permute(0, 2, 1))
print(x.permute(0, 2, 1).shape)

x = torch.arange(10)  # [10]
print('x shape', x.shape)

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x)
print('squeeze - unsqueeze')
print(x.shape)
print(x.squeeze(1).shape)
