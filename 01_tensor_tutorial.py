import torch
import numpy as np

# Initialize a tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# "Uncomment" to run code
# print(f"x_data Tensor: \n{x_data}\n")

# Initialize a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# "Uncomment" to run code
# print(f"x_np Tensor: \n{x_np}\n")

# Create a tensor of ones with the same shape as x_data
x_ones = torch.ones_like(x_data)  # Retains the properties (shape) of x_data

# "Uncomment" to run code
# print(f"Ones Tensor: \n{x_ones}\n")

# Create a tensor with random values and override the datatype to float
x_rand = torch.rand_like(x_data, dtype=torch.float)

# "Uncomment" to run code
# print(f"Random Tensor: \n{x_rand}\n")

# With random or constant values:

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# "Uncomment" to run code
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3, 4)

# "Uncomment" to run code
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# Tensor Operations

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  # print(f"Device tensor is stored on: {tensor.device}")

# Standard numpy-like indexing and slicing:

tensor = torch.ones(4, 4)
tensor[:,1] = 0
# print(tensor)

# Joining Tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# Multiplying tensors

# This computes the element-wise product
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
# print(f"tensor * tensor \n {tensor * tensor}")

# This computes the matrix multiplication between two tensors
# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.

# print(tensor, "\n")
tensor.add_(5)
# print(tensor)

# In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

# Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

# Tensor to NumPy array

t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.

t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

# NumPy array to Tensor

n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.

np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")