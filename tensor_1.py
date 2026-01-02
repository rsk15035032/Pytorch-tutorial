import torch


#======================================================#
#          Initialising Tensor                         #
#======================================================#

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float32, device= device,
                         requires_grad= True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Others common initialization methods
x = torch.empty(size = (3, 3)) # Create a empty matrix filled with zeroes
x = torch.zeros((3, 3))  # Same as above one.
x = torch.rand((3, 3)) # Create a random matrix
x = torch.ones((3, 3)) # Create a matrix all the values with 1.
x = torch.eye(5, 5) # create a matrix in whcih dailonal value 1 and rest all are zeroes.
x = torch.arange(start = 0, end = 5, step = 1) # Create a tensor [0,1,2,3,4]
x = torch.linspace(start= 0.1, end=1, steps=10) #  Create a tensor [0.1,0.2,.......,1] upto 10 steps
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # Create tensor with normal distribution of mean 0 and std 1
x = torch.empty(size=(1, 5)).uniform_(0, 1)# same as previous one for uniform distribution
x = torch.diag(torch.ones(3)) # Create a matrix with daigonal value 1 and rest all 0.

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)

print(tensor.bool())# boolean True/False
print(tensor.short())# int16
print(tensor.long())# int64
print(tensor.half())# float16
print(tensor.float())#float32**
print(tensor.double())#float64

# Array to tensor conversion and vice-versa.
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array) # convert numpy array to tensor
backnp_array = tensor.numpy()
print(backnp_array)