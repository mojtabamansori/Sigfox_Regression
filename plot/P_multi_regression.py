import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from vincenty import vincenty
from scipy.spatial import distance
from sklearn.metrics import r2_score



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

for ii, i in enumerate(range(len(k2)-1)):
    print('1', ii / len(k2)-1)

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

max_x = np.max(Y_train_combined[:, 0])
min_x = np.min(Y_train_combined[:, 0])
max_y = np.max(Y_train_combined[:, 1])
min_y = np.min(Y_train_combined[:, 1])


index_1 = ((min_x < Y_train_combined[:, 0]) &
                             (Y_train_combined[:, 0] < (min_x + ((max_x-min_x)/2))) &
                             (min_y < Y_train_combined[:, 1]) &
                             (Y_train_combined[:, 1] < (min_y + ((max_y-min_y)/3))))
index_2 = (~(((min_x < Y_train_combined[:, 0]) &
                             (Y_train_combined[:, 0] < (min_x + ((max_x-min_x)/2))) &
                             (min_y < Y_train_combined[:, 1]) &
                             (Y_train_combined[:, 1] < (min_y + ((max_y-min_y)/3)))) & ((((min_x + ((max_x-min_x)/2))) < Y_train_combined[:, 0]) &
                             (Y_train_combined[:, 0] < max_x ) &
                             (((min_y + ((2*(max_y-min_y))/3))) < Y_train_combined[:, 1]) &
                             (Y_train_combined[:, 1] < max_y))))

index_3 = ((((min_x + ((max_x-min_x)/2))) < Y_train_combined[:, 0]) &
                             (Y_train_combined[:, 0] < max_x ) &
                             (((min_y + ((2*(max_y-min_y))/3))) < Y_train_combined[:, 1]) &
                             (Y_train_combined[:, 1] < max_y))

n = 100
X_train_combined = preprocess_data(X_train_combined, n)
X_test_combined = preprocess_data(X_test_combined, n)

X_Train_1 = X_train_combined[index_1]
Y_Train_1 = Y_train_combined[index_1]

X_Train_2 = X_train_combined[index_2]
Y_Train_2 = Y_train_combined[index_2]

X_Train_3 = X_train_combined[index_3]
Y_Train_3 = Y_train_combined[index_3]


model1 = RandomForestRegressor()
model1.fit(X_Train_1, Y_Train_1)
model2 = RandomForestRegressor()
model2.fit(X_Train_2, Y_Train_2)
model3 = RandomForestRegressor()
model3.fit(X_Train_3, Y_Train_3)

mean_xtrain1 = np.mean(X_Train_1, axis=0)
mean_xtrain2 = np.mean(X_Train_2, axis=0)
mean_xtrain3 = np.mean(X_Train_3, axis=0)
errors_eval = []

for i in range(len(X_test_combined)):
    print(i/len(X_test_combined))
    X_test_combined[i, :]
    dist1 = distance.euclidean(X_test_combined[i], mean_xtrain1)
    dist2 = distance.euclidean(X_test_combined[i], mean_xtrain2)
    dist3 = distance.euclidean(X_test_combined[i], mean_xtrain3)
    k = np.array([dist1, dist2, dist3])
    k_i = np.argmin(k)

    if k_i == 0:
        pred1 = model1.predict(X_test_combined[i, :].reshape(1,-1))
        errors_eval.append(vincenty(pred1[0], Y_test_combined[i,:]))

    if k_i == 1:
        pred2 = model2.predict(X_test_combined[i, :].reshape(1,-1))
        errors_eval.append(vincenty(pred2[0], Y_test_combined[i, :]))

    if k_i == 2:
        pred3 = model3.predict(X_test_combined[i, :].reshape(1,-1))
        errors_eval.append(vincenty(pred3[0], Y_test_combined[i, :]))

mean_error = np.mean(errors_eval) * 1000
median_error = np.median(errors_eval) * 1000

print(f"Mean Error: {mean_error} meters")
print(f"Median Error: {median_error} meters")


