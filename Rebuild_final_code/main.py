import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def section_true(section_list, Y_true, list_mearge):
    a = list_mearge[1] - 1
    b = list_mearge[2] - 1
    index_a = (((np.array(section_list[a])[0]) < (Y_true[:, 1])) & ((Y_true[:, 1]) < (np.array(section_list[a])[1])))
    index_b = (((np.array(section_list[b])[0]) < (Y_true[:, 1])) & ((Y_true[:, 1]) < (np.array(section_list[b])[1])))
    return (index_a | index_b)


def section_true_not_mearge(section_list, Y_true, i_model):
    if (i_model != 0) and (i_model != 8):
        a = i_model
        index_a = (((np.array(section_list[a])[0]) < (Y_true[:, 1])) & (
                    (Y_true[:, 1]) < (np.array(section_list[a])[1])))
        return index_a


def load_date_def(list_random_seed, n_s):
    dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
    X, Y = dataset[:, :137], dataset[:, 138:]

    X_train_combined = None
    Y_train_combined = None
    X_test_combined = None
    Y_test_combined = None
    flag = 1
    for section in range(n_s):
        index_Y = Y[:, 1]
        Max_getway = np.max(np.max(index_Y))
        min_getway = np.min(np.min(index_Y))
        step = (Max_getway - min_getway) / n_s
        index = (((min_getway + (step * section)) < index_Y) & ((min_getway + (step * (section + 1))) > index_Y))

        X_current = X[index, :]
        Y_current = Y[index, :]

        X_train_temp, X_test_temp, \
            Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current,
                                                         test_size=0.3,
                                                         random_state=list_random_seed)

        imputer = SimpleImputer(strategy='mean')
        X_train_temp_imputed = imputer.fit_transform(X_train_temp)
        X_test_temp_imputed = imputer.transform(X_test_temp)

        if flag == 1:
            if X_train_combined is None:
                X_train_combined = X_train_temp_imputed
                Y_train_combined = Y_train_temp
                X_test_combined = X_test_temp_imputed
                Y_test_combined = Y_test_temp
                flag = 0
        if flag == 0:
            X_train_combined = np.concatenate((X_train_temp_imputed, X_train_combined), axis=0)
            Y_train_combined = np.concatenate((Y_train_temp, Y_train_combined), axis=0)
            X_test_combined = np.concatenate((X_test_temp_imputed, X_test_combined), axis=0)
            Y_test_combined = np.concatenate((Y_test_temp, Y_test_combined), axis=0)

    return X_train_combined, Y_train_combined, X_test_combined, Y_test_combined


def index_section(numebers_section, Y_train_combined, i_model, index, section_list):
    index_Y = Y_train_combined[:, 1]
    index['model_0'] = ((3.0 < index_Y) & (5.0 > index_Y))
    for i_model in range(0, numebers_section):
        index['model_' + str(i_model + 1)] = (
                    ((section_list[i_model, 0]) < index_Y) & ((section_list[i_model, 1]) > index_Y))
    return index


def return_section_list(numebers_section, Max_getway, min_getway):
    section_list = np.zeros((numebers_section - 1, 2))
    for li, section in enumerate(range(numebers_section - 1)):
        step = (Max_getway - min_getway) / numebers_section
        if li == 0:
            section_list[li, 0] = (min_getway + (step * section))
        else:
            section_list[li - 1, 1] = (min_getway + (step * section))
            section_list[li, 0] = (min_getway + (step * section))
        section_list[(numebers_section - 2), 1] = Max_getway
    return section_list


def list_change_section_r1(lists_old, number_section_old, i_1, k, section_list_old):
    res = np.zeros((number_section_old, number_section_old))
    for i in range(1, number_section_old):
        for j in range(i + 1, number_section_old):
            number_multiply = np.intersect1d(lists_old[f"list_{i}"], lists_old[f"list_{j}"])
            res[i, j] = len(number_multiply)
    if 1 < np.max(res):
        return 0
    else:
        return 5


def rename_list(list_with_name_old):
    list_with_name_new = {}
    list_0 = list_with_name_old.pop('list_0')

    for i, key in enumerate(list_with_name_old.keys(), start=1):
        list_with_name_new[f'list_{i}'] = list_with_name_old[key]

    list_with_name_new['list_0'] = list_0
    return list_with_name_new


def list_change_section(lists_old, number_section_old, i_1, k, section_list_old):
    res = np.zeros((number_section_old, number_section_old))
    for i in range(1, number_section_old):
        for j in range(i + 1, number_section_old):
            number_multiply = np.intersect1d(lists_old[f"list_{i}"], lists_old[f"list_{j}"])
            res[i, j] = len(number_multiply)

    a = np.unravel_index(np.argmax(res), res.shape)
    common_elements = np.intersect1d(lists_old[f"list_{a[0]}"], lists_old[f"list_{a[1]}"])
    lists_old.pop(f"list_{a[0]}")
    lists_old.pop(f"list_{a[1]}")

    section_list_new = np.copy(section_list_old)
    temp = len(section_list_new) - 1
    a_index_new = section_list_new[a[0], 0]
    b_index_new = section_list_new[a[1], 1]
    section_list_new = np.delete(section_list_new, a[0], 0)
    section_list_new = np.delete(section_list_new, (a[1] - 1), 0)
    kss = np.zeros((temp, 2))
    lists_old[f"list_{a[0]}_{a[1]}"] = common_elements
    lists_old = rename_list(lists_old)
    kss[0:(temp - 1), :] = section_list_new
    kss[temp - 1, 0] = min(a_index_new, b_index_new)
    kss[temp - 1, 1] = max(a_index_new, b_index_new)

    k.append(i_1)
    k.append(a[0])
    k.append(a[1])

    return lists_old, (number_section_old - 1), k, kss


def f_e_mean_std(input_model, output_model, n_s):
    data_plot_mean = np.zeros((n_s, 137))
    data_plot_std = np.zeros((n_s, 137))
    for section in range(n_s):
        index_Y = output_model[:, 1]
        Max_getway = np.max(np.max(index_Y))
        min_getway = np.min(np.min(index_Y))
        step = (Max_getway - min_getway) / n_s
        data_getway = input_model[
            (((min_getway + (step * section)) < index_Y) & ((min_getway + (step * (section + 1))) > index_Y))]
        mean_getway = np.mean(data_getway, axis=0)
        std_getway = np.std(np.float32(data_getway), axis=0)
        data_plot_mean[section, :] = mean_getway
        data_plot_std[section, :] = std_getway

    getway_useful = []
    for number_getway in range(137):
        for number_sections in range(n_s):
            if data_plot_mean[number_sections, number_getway] == -200:
                if data_plot_std[number_sections, number_getway] != 0:
                    getway_useful.append(number_sections)
                    getway_useful.append(number_getway)

            if data_plot_mean[number_sections, number_getway] != -200:
                getway_useful.append(number_sections)
                getway_useful.append(number_getway)
    return getway_useful


def list_getways(useful_getways, X_train_combined, i_model):
    X_train_combine_model = X_train_combined[:, useful_getways]
    X_train_combine_model = X_train_combine_model[:, np.array([i_model])]
    return X_train_combine_model


def initialize_lists(n):
    return {f'list_{i}': [] for i in range(n)}


def main():
    random_seeds = 42
    n_sections = 10
    errors = []

    for seed in random_seeds:
        X_train_combined, Y_train_combined, X_test_combined, Y_test_combined = load_date_def([seed], n_sections)
        number_section_old = 9
        Max_getway = np.max(Y_train_combined[:, 1])
        min_getway = np.min(Y_train_combined[:, 1])
        section_list_old = return_section_list(number_section_old, Max_getway, min_getway)
        index = index_section(number_section_old, Y_train_combined, 1, {}, section_list_old)
        useful_getways = f_e_mean_std(X_train_combined, Y_train_combined, n_sections)
        lists = initialize_lists(10)

        for i_model in range(number_section_old):
            X_Train_combine_model = list_getways(useful_getways, X_train_combined, i_model)

            if len(X_Train_combine_model) != 0:
                model = RandomForestRegressor(random_state=seed)
                model.fit(X_Train_combine_model, Y_train_combined)
                pred = model.predict(X_test_combined)

                error = []
                for range_longitude in range(len(pred)):
                    centroids = pred[range_longitude]
                    error.append(vincenty(centroids, Y_test_combined[range_longitude]))
                errors.append(np.mean(error))

        for i in range(100):
            lists, number_section_old, k, section_list_old = list_change_section(lists, number_section_old, i, [],
                                                                                 section_list_old)
            if number_section_old == 5:
                break

    print("Errors: ", errors)


if __name__ == "__main__":
    main()
