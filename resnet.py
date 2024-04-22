import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import haversine_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()

# Define the range for file names
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]

# Load data and concatenate
for ii, i in enumerate(k):
    file_name = f"session/data_{i:.1f}_to_{i + 0.1:.1f}.csv"
    df = pd.read_csv(file_name)
    data_array = df.to_numpy()
    X_current = data_array[:, :137]
    Y_current = data_array[:, 137:]

    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3,
                                                                            random_state=42)
    if ii == 0:
        X_train_combined = X_train_temp
        Y_train_combined = Y_train_temp
        X_test_combined = X_test_temp
        Y_test_combined = Y_test_temp

    else:
        X_train_combined = np.concatenate((X_train_temp, X_train_combined), axis=0)
        Y_train_combined = np.concatenate((Y_train_temp, Y_train_combined), axis=0)
        X_test_combined = np.concatenate((X_test_temp, X_test_combined), axis=0)
        Y_test_combined = np.concatenate((Y_test_temp, Y_test_combined), axis=0)


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
y_train_tensor = torch.tensor(Y_train_combined, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)

# Define hyperparameters
input_size = X_train_combined.shape[1]
output_size = Y_train_combined.shape[1]
learning_rate = 0.001
num_epochs = 5
batch_size = 32

# Create DataLoader for training set
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = NeuralNetwork(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.numpy()

# Calculate errors using Hausdorff distance
errors = []
print("Real data shape:", Y_test_combined.shape)
print("Predicted data shape:", y_pred.shape)

for i in range(len(Y_test_combined)):
    print()
    error = haversine_distances(np.reshape(np.radians(Y_test_combined[i]), (1, -1)), np.reshape(np.radians(y_pred), (1, -1)))* 6371000
    errors.append(error)
errors = np.array(errors)

print(f"Mean Error: {np.mean(errors)}")
print(f"Median Error: {np.median(errors)}")

# Plot histogram of errors
plt.hist(errors, bins=100)
plt.xlabel('Error (meters)')
plt.ylabel('Frequency')
plt.title('Histogram of Errors')
plt.show()
