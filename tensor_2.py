import torch

#======================================================#
#          Tensor maths and Comparison Operation       #
#======================================================#

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out= z1)

z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

#Inplace Operation
t = torch.zeros(3)
t.add_(x)
t += x # t = t + x not inplace operation it will make copy first.

# Exponentiation
z = x.pow(2) # method 1
z = x **2  # method 2

# Simple comparison
z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2) # Method 2

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

#element wise mult.
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand(batch, n, m)
tensor2 = torch.rand(batch, m, p)
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)

# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operation
sum_x = torch.sum(x, dim=0) # for summation where dim=0 becasue vector 
values, indices = torch.max(x, dim=0) # max value
values, indices = torch.min(x, dim=0) # Min value
abs_x = torch.abs(x) # Absolute value
z = torch.argmax(x, dim=0) 
z = torch.argmax(x, dim=0)
mean_x = torch.mean(x.float(), dim=0) # mean
z = torch.eq(x, y) # check it is equal or not
sorted_y, indices = torch.sort(y, dim=0, descending=False) # sorting

z = torch.clamp(x, min=0) # any value smaller than 0 will set to 0

x = torch.tensor([1,0,1,1,1,1], dtype=torch.bool)
z = torch.any(x) # True
z = torch.all(x) # False







