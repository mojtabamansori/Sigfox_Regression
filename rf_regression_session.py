import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty
from sklearn.metrics import mean_absolute_error, r2_score

def preproces(x,number):
    if number == 0:
        X_current = (x - np.min(x)) / np.min(x) * -1
    if number == 1:
        X_current = np.exp((x - np.min(x))/24)/np.exp(np.min(x)*-1/24)
    if number == 2:
        X_current = x ** np.e
    return X_current


def evalution(Y_test_combined, pred,i2):
    errors = []
    for range_longtiude in range(len(pred)):
        centroids = pred[range_longtiude]
        error = vincenty(centroids, Y_test_combined[range_longtiude])
        errors.append(error)

    print(f"{i2}_Mean Error: {np.mean(errors) * 1000} meters")
    print(f"{i2}_Median Error: {np.median(errors) * 1000} meters")
    print(f"{i2}_R2 Score: {r2_score(Y_test_combined, pred)}\n")

    mean_error = np.mean(errors) * 1000
    median_error = np.median(errors) * 1000
    R2_score = r2_score(Y_test_combined, pred)

    results_df = pd.DataFrame({
        'Range Longitude': i2,
        'Mean Error (meters)': [mean_error] ,
        'Median Error (meters)': [median_error],
        'R2 Score': [R2_score]
    })

    # Save results to CSV
    results_df.to_csv('evaluation_results.csv', index=False)

k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]
for i2 in range(50,60,1):
    regressor = RandomForestRegressor()
    X_train_combined, Y_train_combined = [], []
    X_test_combined, Y_test_combined = [], []
    for range_longtiude in k:
        file_name = f"session/data_{range_longtiude:.1f}_to_{range_longtiude + 0.1:.1f}.csv"
        df = pd.read_csv(file_name)
        data_array = df.to_numpy()
        X_current = data_array[:, :137]
        Y_current = data_array[:, 137:]
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3, random_state=i2)
        X_train_combined.append(X_train_temp)
        Y_train_combined.append(Y_train_temp)
        X_test_combined.append(X_test_temp)
        Y_test_combined.append(Y_test_temp)

    X_train_combined = np.concatenate(X_train_combined, axis=0)
    Y_train_combined = np.concatenate(Y_train_combined, axis=0)
    X_test_combined = np.concatenate(X_test_combined, axis=0)
    Y_test_combined = np.concatenate(Y_test_combined, axis=0)

    regressor.fit(X_train_combined, Y_train_combined)
    pred = regressor.predict(X_test_combined)

    evalution(Y_test_combined,pred,i2)

