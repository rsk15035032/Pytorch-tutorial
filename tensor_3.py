import torch

#======================================================#
#                Tensor Indexing                       #
#======================================================#

batch_size = 10
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape) # x[0,:]
print(x[:, 0].shape) # Printing all features of 1st row.
print(x[2, 0:10]) # 0:10 ---> [0, 1, 2,....., 9].

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand(3, 5)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])# return the element less than 2 and greater than 8.
print(x[x.remainder(2) == 0]) # return all the even number.

# Useful operation
print(torch.where(x > 5, x, x*2)) # return multiple of 2 and number greater than 5.

