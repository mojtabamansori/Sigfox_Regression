import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty


def preproces(x, number):
    x_current = None
    if number == 0:
        x_current = (x - np.min(x)) / np.min(x) * -1
    elif number == 1:
        x_current = np.exp((x - np.min(x)) / 24) / np.exp(np.min(x) * -1 / 24)
    elif number == 2:
        x_current = (x - np.min(x)) / np.min(x) * -1
        x_current = x_current ** np.e
    return x_current, number


def preprocess_data(X, number_selected_feature):
    x_processed = np.zeros((X.shape[0], number_selected_feature * 2 + 1))
    for row_number in range(X.shape[0]):
        temp = X[row_number, :]
        largest_n, max_columns, num_neg_200 = feature_extraction(temp, number_selected_feature)
        x_processed[row_number, :number_selected_feature] = largest_n[0]
        x_processed[row_number, number_selected_feature:number_selected_feature * 2] = max_columns
        x_processed[row_number, number_selected_feature * 2] = num_neg_200
    return x_processed


def evaluation(y_test_eval, y_hat, i2, number):
    errors_eval = []
    for range_longitude in range(len(y_hat)):
        centroids = y_hat[range_longitude]
        error = vincenty(centroids, y_test_eval[range_longitude])
        errors_eval.append(error)

    mean_error = np.mean(errors_eval) * 1000
    median_error = np.median(errors_eval) * 1000
    r_2_score = r2_score(y_test_eval, y_hat)

    print(f"Mean Error: {mean_error} meters")
    print(f"Median Error: {median_error} meters")
    print(f"R2 Score: {r_2_score}\n")

    results_df = pd.DataFrame({
        'Random': i2,
        'Mean Error (meters)': [mean_error],
        'Median Error (meters)': [median_error],
        'R2 Score': [r_2_score],
        'Pre process': number
    })
    results_df.to_csv(f'result/evaluation_results_{number}_{i2}.csv', index=False)


def feature_extraction(matrix, number_feature):
    flattened_matrix = matrix[matrix != -200]
    if len(flattened_matrix) < number_feature:
        largest_n = flattened_matrix
    else:
        largest_n = np.partition(flattened_matrix, -number_feature)[-number_feature:]
    max_columns = np.argsort(matrix)[-number_feature:]
    num_neg_200 = np.sum(matrix == -200)
    return largest_n, max_columns, num_neg_200


X_train_combined = None
Y_train_combined = None
X_test_combined = None
Y_test_combined = None


for N_feature in range(1, 20):
    k2 = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]
    for ks in range(35, 42):
        for ii, i in enumerate(k2):
            file_name = f"session/data_{i:.1f}_to_{i + 0.1:.1f}.csv"
            df = pd.read_csv(file_name)
            data_array = df.to_numpy()
            X_current = data_array[:, :137]
            Y_current = data_array[:, 137:]

            X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current,
                                                                                    test_size=0.3, random_state=ks)
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
        errors = [0]
        X_train = X_train_combined
        y_train = Y_train_combined
        X_test = X_test_combined
        y_test = Y_test_combined

        n = N_feature
        X_train = preprocess_data(X_train, n)
        X_test = preprocess_data(X_test, n)
        print(f'-------------------- {n}_{ks} -----------------------')
        regressor = RandomForestRegressor()
        for i_pre in [0, 1, 2]:
            X_train_combined_p, _ = preproces(X_train_combined, i_pre)
            X_test_combined_p, _ = preproces(X_test_combined, i_pre)

            regressor.fit(X_train_combined_p, Y_train_combined)
            pred = regressor.predict(X_test_combined_p)
            evaluation(Y_test_combined, pred, 1, i_pre)
