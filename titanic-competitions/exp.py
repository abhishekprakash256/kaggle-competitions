
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

print(X.shape)
print(y.shape)


import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)   # Input layer to hidden layer
        self.fc2 = nn.Linear(64, 1)            # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()            # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))            # Apply ReLU activation to the hidden layer
        x = self.fc2(x)                        # Linear transformation to output layer
        x = self.sigmoid(x)                    # Sigmoid activation for binary classification
        return x


# Create an instance of the BinaryClassifier model
input_size = 3
model = BinaryClassifier(input_size)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y.unsqueeze(1).float())  # Ensure the label shape matches the prediction shape

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at regular intervals
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


