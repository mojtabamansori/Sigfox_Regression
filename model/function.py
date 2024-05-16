import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from vincenty import vincenty

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
        for number_sections in range(n_s):
            if data_plot_mean[number_sections, number_getway] != -200:
                getway_useful.append(number_sections)
                getway_useful.append(number_getway)
    return getway_useful


def list_getways(useful_section_getway, n_s):
    lists = {}
    for i in range(n_s):
        lists['list_' + str(i+1)] = []
    for name_section, gateway in zip(useful_section_getway[0::2], useful_section_getway[1::2]):
        lists[f"list_{name_section+1}"].append(gateway)

    lists["list_0"] = [9, 10, 11, 12, 17,
                  19, 20, 22, 26, 30,
                  58, 61, 66, 70, 71,
                  72, 75, 82, 83, 84,
                  85, 86, 88, 89, 90, 91,
                  92, 94, 96, 97, 99, 100,
                  101, 103, 104, 105, 107,
                  110, 118, 119, 28, 24, 18,
                  62, 102, 126, 0, 1, 2, 4,
                  6, 7, 8, 13, 14, 15, 16,
                  21, 29, 31, 32, 33,
                  36, 37, 38, 39, 40, 43,
                  44, 59, 60, 64, 68, 73, 109]
    for i in range(n_s+1):
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


def Label_area(pre, n_s, Y_train_combined):
    index = {}
    P = {}
    for section in range(n_s):
        index_Y = pre[:, 1]
        Max_getway = np.max(np.max(index_Y))
        min_getway = np.min(np.min(index_Y))
        step = (Max_getway - min_getway) / n_s
        index['model_' + str(section)] = (((min_getway + (step * section)) < index_Y) & ((min_getway + (step * (section + 1))) > index_Y))
        P['model_' + str(section)] = index['model_' + str(section)].copy()
        P['model_' + str(section)][P['model_' + str(section)]] = section
        P['model_' + str(section)] = np.where(P['model_' + str(section)], section, 0)
    return P



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
