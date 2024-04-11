import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import haversine_distances

# Read the dataset
df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()



# Define the range for file names
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]

for ii,i in enumerate(k):
    file_name = f"session/data_{i:.1f}_to_{i + 0.1:.1f}.csv"
    df = pd.read_csv(file_name)
    data_array = df.to_numpy()
    X_current = data_array[:, :137]
    Y_current = data_array[:, 137:]

    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3, random_state=42)
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
k = 1
errors = []
X_train = X_train_combined
y_train = Y_train_combined
X_test = X_test_combined
y_test = Y_test_combined
print(y_train.shape,X_train.shape)

for n,i in enumerate(range(len(X_test))):
    print(f'\r{n}/{len(X_test)}',end='')
    all_distances = np.sqrt(np.sum(np.abs(X_train - X_test[i]) ** 2, axis=1))

    k_indexes = np.argsort(all_distances)[0:k]
    centroids = np.mean(y_train[k_indexes, :], axis=0)
    error = haversine_distances(np.reshape(np.radians(y_test[i]), (1, -1)), np.reshape(np.radians(centroids), (1, -1)))* 6371000
    errors.append(error)


print(np.mean(errors))
print(np.median(errors))