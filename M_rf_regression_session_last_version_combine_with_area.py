from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty


def preproces(x, number):
    X_current = None
    if number == 0:
        X_current = (x - np.min(x)) / np.min(x) * -1
    elif number == 1:
        X_current = np.exp((x - np.min(x)) / 24) / np.exp(np.min(x) * -1 / 24)
    elif number == 2:
        X_current = (x - np.min(x)) / np.min(x) * -1
        X_current = X_current ** np.e
    return X_current, number


def Label_area(pre):
    pr = pre[:, 1].copy()
    for i in range(len(pr)):
        if pr[i] < 3.9:
            pr[i] = 0
        if ((pr[i] > 3.9) & (pr[i] < 4.1)):
            pr[i] = 1
        if pr[i] > 4.1:
            pr[i] = 2
    return pr


def list_to_data(list, X_train_combined, X_test_combined):
    X_Train_1 = None
    X_test_1 = None
    for iii, number_col in enumerate(list):
        if iii == 0:
            X_Train_1 = X_train_combined[:, number_col].reshape(-1, 1)
            X_test_1 = X_test_combined[:, number_col].reshape(-1, 1)
        else:
            X_Train_1 = np.concatenate((X_Train_1, X_train_combined[:, number_col].reshape(-1, 1)), axis=1)
            X_test_1 = np.concatenate((X_test_1, X_test_combined[:, number_col].reshape(-1, 1)), axis=1)
    return X_Train_1, X_test_1

def evaluation(Y_test_combined, pred, i2, number):
    errors = []
    for range_longitude in range(len(pred)):
        centroids = pred[range_longitude]
        error = vincenty(centroids, Y_test_combined[range_longitude])
        errors.append(error)

    mean_error = np.mean(errors) * 1000
    median_error = np.median(errors) * 1000
    R2_score = r2_score(Y_test_combined, pred)

    print(f"i_pre {i_pre}:randomseed_{i2}_Mean Error: {mean_error} meters")
    print(f"i_pre {i_pre}_randomseed_{i2}_Median Error: {median_error} meters")
    print(f"i_pre {i_pre}_randomseed_{i2}_R2 Score: {R2_score}\n")

    results_df = pd.DataFrame({
        'Random': i2,
        'Mean Error (meters)': [mean_error],
        'Median Error (meters)': [median_error],
        'R2 Score': [R2_score],
        'Pre process': number
    })

    # Save results to CSV
    results_df.to_csv(f'result/evaluation_results_{number}_{i2}.csv', index=False)

list_1 = [9, 10, 11, 12, 17, 19, 20, 22, 26, 30, 58, 61, 66, 70, 71, 72, 75, 82, 83, 84, 85,86, 88, 89, 90, 91 ,
          92, 94, 96, 97, 99, 100, 101, 103, 104, 105, 107, 110, 118, 119,3 , 5, 23, 25, 27, 67]
list_2 = [28, 24, 18, 62, 102, 126, 3, 5, 23, 25, 27, 67]
list_3 = [0, 1, 2, 4, 6, 7, 8, 13, 14, 15, 16, 21, 29, 31, 32, 33, 36, 37, 38, 39, 40, 43, 44, 59, 60, 64, 68, 73, 109,3, 5, 23, 25, 27, 67]


list_fualt_not = [9, 10, 11, 12, 17, 19, 20, 22,
                  26, 30, 58, 61, 66, 70,
                  71, 72, 75, 82, 83, 84, 85,86,
                  88, 89, 90, 91, 92,
                  94, 96, 97, 99, 100, 101, 103,
                  104, 105, 107, 110, 118, 119,
                  28, 24, 18, 62, 102, 126,
                  0, 1, 2, 4, 6, 7, 8, 13, 14,
                  15, 16, 21, 29, 31, 32, 33, 36,
                  37, 38, 39, 40, 43, 44, 59, 60, 64, 68, 73, 109]

k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2]
for random_seed_number in range(40, 60, 1):
    regressor = RandomForestRegressor()
    regressor_1 = RandomForestRegressor()
    regressor_2 = RandomForestRegressor()
    regressor_3 = RandomForestRegressor()
    X_train_combined, Y_train_combined = [], []
    X_test_combined, Y_test_combined = [], []
    for range_longitude in k:
        file_name = f"session/data_{range_longitude:.1f}_to_{(range_longitude + 0.1):.1f}.csv"
        df = pd.read_csv(file_name)
        data_array = df.to_numpy()
        X_current = data_array[:, :137]
        Y_current = data_array[:, 137:]
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current, test_size=0.3,
                                                                                random_state=random_seed_number)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train_temp_imputed = imputer.fit_transform(X_train_temp)
        X_test_temp_imputed = imputer.transform(X_test_temp)

        X_train_combined.append(X_train_temp_imputed)
        Y_train_combined.append(Y_train_temp)
        X_test_combined.append(X_test_temp_imputed)
        Y_test_combined.append(Y_test_temp)

    X_train_combined = np.concatenate(X_train_combined, axis=0)
    Y_train_combined = np.concatenate(Y_train_combined, axis=0)
    X_test_combined = np.concatenate(X_test_combined, axis=0)
    Y_test_combined = np.concatenate(Y_test_combined, axis=0)

    X_train_combined_all, X_test_combined_all = list_to_data(list_fualt_not, X_train_combined, X_test_combined)
    X_train_combined_1, X_test_combined_1 = list_to_data(list_1, X_train_combined, X_test_combined)
    X_train_combined_2, X_test_combined_2 = list_to_data(list_2, X_train_combined, X_test_combined)
    X_train_combined_3, X_test_combined_3 = list_to_data(list_3, X_train_combined, X_test_combined)

    for i_pre in [0, 1, 2]:
        X_train_combined_p, t = preproces(X_train_combined_all, i_pre)
        X_train_combined_p_1, t = preproces(X_train_combined_1, i_pre)
        X_train_combined_p_2, t = preproces(X_train_combined_2, i_pre)
        X_train_combined_p_3, t = preproces(X_train_combined_3, i_pre)

        X_test_combined_p, t = preproces(X_test_combined_all, i_pre)
        X_test_combined_p_1, t = preproces(X_test_combined_1, i_pre)
        X_test_combined_p_2, t = preproces(X_test_combined_2, i_pre)
        X_test_combined_p_3, t = preproces(X_test_combined_3, i_pre)

        regressor.fit(X_train_combined_p, Y_train_combined)
        pred = regressor.predict(X_test_combined_p)

        regressor_1.fit(X_train_combined_p_1, Y_train_combined)
        regressor_2.fit(X_train_combined_p_2, Y_train_combined)
        regressor_3.fit(X_train_combined_p_3, Y_train_combined)

        pred_1 = regressor_1.predict(X_test_combined_p_1)
        pred_2 = regressor_2.predict(X_test_combined_p_2)
        pred_3 = regressor_3.predict(X_test_combined_p_3)

        labe_are = Label_area(pred)
        for i_number_label in range(len(X_test_combined)):
            if labe_are[i_number_label] == 0:
                pred[i_number_label, :] = pred_1[i_number_label, :]
            if labe_are[i_number_label] == 1:
                pred[i_number_label, :] = pred_2[i_number_label, :]
            if labe_are[i_number_label] == 2:
                pred[i_number_label, :] = pred_3[i_number_label, :]


        evaluation(Y_test_combined, pred, random_seed_number, i_pre)
