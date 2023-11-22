
import numpy as np
import torch

# Define the size of your dataset
num_samples = 1000

# Create dummy data with 3 input features
X = np.random.rand(num_samples, 3).astype(np.float32)

# Generate dummy labels for binary classification (0 or 1)
y = np.random.randint(2, size=num_samples)

# Convert the NumPy arrays to PyTorch tensors
X = torch.tensor(X)
y = torch.tensor(y)