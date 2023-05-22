import numpy as np
import torch

# Example list of numpy arrays
# list_of_arrays = [np.random.randn(3, 4) for _ in range(5)]

# Concatenate the arrays into a single numpy array
# concatenated_array = np.concatenate(list_of_arrays)

# Convert the concatenated array to a PyTorch tensor
tensor = torch.zeros((2,2)+(2,))
tensor[1]=torch.from_numpy(np.random.randn(2,2))
print(tensor)