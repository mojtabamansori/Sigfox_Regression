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

df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()
X_Train = data_array[:, :137]
Y_Train = data_array[:, 138:]

k2 = [3.59, 3.7137,
      3.7138,
      3.95,
      4.13305,
      4.1333,
      4.15,
      4.36]
k2 = np.array(k2)

for i in range(len(k2)-1):

    indices = np.argwhere((k2[i] <= Y_Train[:, 1]) & (Y_Train[:, 1] <= k2[i+1]))

    filtered_X_Train = X_Train[indices, :]
    filtered_Y_Train = Y_Train[indices, :]

    filtered_X_Train = np.reshape(filtered_X_Train, (filtered_X_Train.shape[0], filtered_X_Train.shape[2]))
    filtered_Y_Train = np.reshape(filtered_Y_Train, (filtered_Y_Train.shape[0], filtered_Y_Train.shape[2]))

    filtered_data = np.concatenate((filtered_X_Train, filtered_Y_Train), axis=1)

    X_current = filtered_data[:, :137]
    Y_current = filtered_data[:, 137:]


    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current,
                                                                            test_size=0.3, random_state=42)


    area_moshtrak_index = np.zeros((8, 137))

    # print(X_train_temp.shape())
    X_train_temp = np.array(X_train_temp)
    Y_train_temp = np.array(Y_train_temp)
    session_X_test = np.array(X_test_temp)
    session_Y_test = np.array(Y_test_temp)

    indices = np.argwhere((k2[i] <= Y_train_temp[:, 1]) & (Y_train_temp[:, 1] <= k2[i + 1]))

    session_X_train = X_train_temp[indices[:, 0], :]
    session_Y_train = Y_train_temp[indices[:, 0], :]
    # common_pattern = area_moshtrak_index[ii]
    model = RandomForestRegressor()
    model.fit(session_X_train, session_Y_train)
    y_pred = model.predict(session_X_test)

    if np.all(session_X_train == -200) == np.all(session_X_test == -200):
        print(len(y_pred))
        evaluation(session_Y_test, y_pred, 1, number=42)
