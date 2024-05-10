import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from vincenty import vincenty  # Importing Vincenty formula for distance calculation


# Read the dataset
df = pd.read_csv("sigfox_dataset_rural (1).csv")

# Define the range for file names
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]

# Define the RandomForestRegressor
regressor = RandomForestRegressor()

# Initialize lists to store combined data
X_train_combined, Y_train_combined = [], []
X_test_combined, Y_test_combined = [], []

for i in k:
    file_name = f"session/data_{i:.1f}_to_{i + 0.1:.1f}.csv"
    df = pd.read_csv(file_name)
    data_array = df.to_numpy()
    X_current = data_array[:, :137]
    Y_current = data_array[:, 137:]

    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3, random_state=42)

    X_train_combined.append(X_train_temp)
    Y_train_combined.append(Y_train_temp)
    X_test_combined.append(X_test_temp)
    Y_test_combined.append(Y_test_temp)

# Concatenate the lists to form combined datasets
X_train_combined = np.concatenate(X_train_combined, axis=0)
Y_train_combined = np.concatenate(Y_train_combined, axis=0)
X_test_combined = np.concatenate(X_test_combined, axis=0)
Y_test_combined = np.concatenate(Y_test_combined, axis=0)

# Train the RandomForestRegressor on the combined training data
regressor.fit(X_train_combined, Y_train_combined)

# Predict on the combined test data
pred = regressor.predict(X_test_combined)

# Calculate errors using Vincenty formula
errors = []
for i in range(len(pred)):
    centroids = pred[i]
    error = vincenty(centroids, Y_test_combined[i]) * 1000  # Using Vincenty formula and converting to meters
    errors.append(error)

# Print mean and median errors
print(f"Mean Error: {np.mean(errors)} meters")
print(f"Median Error: {np.median(errors)} meters")
r2 = r2_score(Y_test_combined, pred)
print(f"R2 Score: {r2}")
