import numpy as np
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty
from sklearn.svm import SVR

# from M_mean_std import  numebers_section
# print(f"numebers_section is {numebers_section}")
def f_e_mean_std(input_model, output_model, n_s):
    data_plot_mean = np.zeros((n_s, 137))
    data_plot_std = np.zeros((n_s, 137))
    for section in range(n_s):
        index_Y = output_model[:, 1]
        Max_getway = np.max(np.max(index_Y))
        min_getway = np.min(np.min(index_Y))
        step = (Max_getway - min_getway) / n_s
        data_getway = input_model[(((min_getway + (step * section)) < index_Y) & ((min_getway + (step * (section + 1))) > index_Y))]
        mean_getway = np.mean(data_getway, axis=0)
        std_getway = np.std(np.float32(data_getway), axis=0)
        data_plot_mean[section, :] = mean_getway
        data_plot_std[section, :] = std_getway


    getway_useful = []
    for number_getway in range(137):
        getway_useful.append(1)
        for number_sections in range(n_s):
            if data_plot_mean[number_sections, number_getway] != -200:
                getway_useful.append(number_sections)
                getway_useful.append(number_getway)
    return getway_useful


def list_getways(useful_section_getway,n_s):
    lists = {}
    flags = {}
    for i in range(n_s):
        lists['list_' + str(i)] = []
        flags['flag_' + str(i)] = 0
        for name_section in useful_section_getway:
            if flags[f'flag_{i}'] == 1:
                lists[f'list_{i}'].append(name_section)
                flags[f'flag_{i}'] = 0
            if name_section == 0:
                flags[f'flag_{i}'] = 1
    for i in range(n_s):
        lists[f"list_{i}"] = np.unique(np.array(lists[f"list_{i}"]))
    return lists

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
        if pr[i] < 3.94:
            pr[i] = 0
        if ((pr[i] > 3.94) & (pr[i] < 4.03)):
            pr[i] = 1
        if pr[i] > 4.03:
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

    # print(f"i_pre {i_pre}:randomseed_{i2}_Mean Error: {mean_error} meters")
    # print(f"i_pre {i_pre}_randomseed_{i2}_Median Error: {median_error} meters")
    # print(f"i_pre {i_pre}_randomseed_{i2}_R2 Score: {R2_score}\n")

    results_df = pd.DataFrame({
        'Random': i2,
        'Mean Error (meters)': [mean_error],
        'Median Error (meters)': [median_error],
        'R2 Score': [R2_score],
        'Pre process': number
    })
    results_df.to_csv(f'../result/evaluation_results_{number}_{i2}.csv', index=False)
    return mean_error

def evaluation1(Y_test_combined, pred, i2, number,i_pre):
    errors = []
    for range_longitude in range(len(pred)):
        centroids = pred[range_longitude]
        error = vincenty(centroids, Y_test_combined[range_longitude])
        errors.append(error)

    mean_error = np.mean(errors) * 1000
    median_error = np.median(errors) * 1000
    # R2_score = r2_score(Y_test_combined, pred)

    print(f"i_pre {i_pre}:randomseed_{i2}_Mean Error: {mean_error} meters")

    results_df = pd.DataFrame({
        'Random': i2,
        'Mean Error (meters)': [mean_error],
        'Median Error (meters)': [median_error],
        # 'R2 Score': [R2_score],
        'Pre process': number
    })
    results_df.to_csv(f'../result/evaluation_results_{number}_{i2}.csv', index=False)
