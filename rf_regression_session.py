import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty
from sklearn.metrics import mean_absolute_error, r2_score

# Define the range for file names
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]
for i2 in range(50,60,1):

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

        X_current = (X_current - np.min(X_current)) / np.min(X_current) * -1
        # X_exp = np.exp((X_current - np.min(X_current))/24)/np.exp(np.min(X_current)*-1/24)
        X_pow = X_current ** np.e

        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3, random_state=i2)

        X_train_combined.append(X_train_temp)
        Y_train_combined.append(Y_train_temp)
        X_test_combined.append(X_test_temp)
        Y_test_combined.append(Y_test_temp)


    # Concatenate the lists to form combined datasets
    X_train_combined = np.concatenate(X_train_combined, axis=0)
    Y_train_combined = np.concatenate(Y_train_combined, axis=0)
    X_test_combined = np.concatenate(X_test_combined, axis=0)
    Y_test_combined = np.concatenate(Y_test_combined, axis=0)

    # dataset = pd.read_csv('sigfox_dataset_rural (1).csv')
    # X = dataset.iloc[:, :137]
    # y = dataset.iloc[:, 138:]
    # X = np.array(X)
    # y = np.array(y)
    # X_norm = (X - np.min(X)) / np.min(X) * -1
    # X_train_combined, X_test_combined, Y_train_combined, Y_test_combined = train_test_split(X, y, test_size=0.3,
    #                                                                         random_state=i2)
    # Train the RandomForestRegressor on the combined training data
    regressor.fit(X_train_combined, Y_train_combined)

    # Predict on the combined test data
    pred = regressor.predict(X_test_combined)

    errors = []
    for i in range(len(pred)):
        centroids = pred[i]
        error = vincenty(centroids, Y_test_combined[i])
        errors.append(error)


    print(f"{i2}_Mean Error: {np.mean(errors)*1000} meters")
    print(f"{i2}_Median Error: {np.median(errors)*1000} meters")
    print(f"{i2}_R2 Score: {r2_score(Y_test_combined, pred)}\n")

