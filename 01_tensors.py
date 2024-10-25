import torch
import numpy as np

# Initialize a tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data Tensor: \n{x_data}\n")

# Initialize a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np Tensor: \n{x_np}\n")

# Create a tensor of ones with the same shape as x_data
x_ones = torch.ones_like(x_data)  # Retains the properties (shape) of x_data
print(f"Ones Tensor: \n{x_ones}\n")

# Create a tensor with random values and override the datatype to float
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n{x_rand}\n")

# With random or constant values:

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
