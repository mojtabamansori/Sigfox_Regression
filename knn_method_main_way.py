import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import haversine_distances
from math import radians

# Load your train and test datasets
train_dataset = pd.read_csv('train_points_with_all_columns.csv')
test_dataset = pd.read_csv('test_points_with_all_columns.csv')

# Assuming your features are in columns 0 to 136
X_train = np.array(train_dataset.iloc[:, :137])
X_test = np.array(test_dataset.iloc[:, :137])

# Assuming your target coordinates (latitude and longitude) are in columns 'Latitude' and 'Longitude'
y_train = np.array(train_dataset[['Latitude', 'Longitude']])
y_test = np.array(test_dataset[['Latitude', 'Longitude']])

k = 1
errors = []

for n,i in enumerate(range(len(X_test))):
    print(f'\r{n}/{len(X_test)}',end='')
    all_distances = np.sqrt(np.sum(np.abs(X_train - X_test[i]) ** 2, axis=1))
    k_indexes = np.argsort(all_distances)[:k]
    centroids = np.mean(y_train[k_indexes, :], axis=0)
    error = haversine_distances(np.reshape(np.radians(y_test[i]), (1, -1)), np.reshape(np.radians(centroids), (1, -1))) * 6371000
    errors.append(error)

print("Mean error:", np.mean(errors))
print("Median error:", np.median(errors))
