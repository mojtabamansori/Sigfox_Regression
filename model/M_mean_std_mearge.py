from function import *
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

numebers_section = 3
last_numebers_section = 2
list_fualt_not = [9, 10, 11, 12, 17,
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
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2]

dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
X, Y = dataset[:, :137], dataset[:, 138:]

section_list = return_section_list(numebers_section, np.max(np.max(Y[:, 1])), np.min(np.min(Y[:, 1])))

useful_section_getway = f_e_mean_std(X, Y, numebers_section)
lists = list_getways(useful_section_getway, numebers_section)

list_gateway_mearge =[]
list_section_data = []
for i in range(20):
    list_section_data.append([])

for number_mearge in range(numebers_section - last_numebers_section):
    lists, numebers_section, list_gateway_mearge = list_change_section(lists, numebers_section, number_mearge, list_gateway_mearge)


for random_seed_number in range(42, 43, 1):
    X_train_combined, Y_train_combined = [], []
    X_test_combined, Y_test_combined = [], []
    for range_longitude in k:
        file_name = f"../session/data_{range_longitude:.1f}_to_{(range_longitude + 0.1):.1f}.csv"
        df = pd.read_csv(file_name)
        data_array = df.to_numpy()
        X_current = data_array[:, :137]
        Y_current = data_array[:, 137:]
        X_train_temp, X_test_temp,\
            Y_train_temp, Y_test_temp = train_test_split(X_current, Y_current,
                                                                    test_size=0.3,
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

Models = {}
Preds = {}
Preds_section = {}
y_section = {}
X_Train_combine = {}
X_Train_combine_p = {}
Y_Train_combine = {}
X_Test_combine = {}
X_Test_combine_p = {}
index = {}

X_train_combined_all, X_test_combined_all = list_to_data(list_fualt_not,
                                                     X_train_combined, X_test_combined)

for i_model in range(numebers_section+1):
    Models['regressor_' + str(i_model)] = RandomForestRegressor()
    Preds['Pred_' + str(i_model)] = None
    Preds_section['Preds_section' + str(i_model)] = None
    y_section['y_section' + str(i_model)] = None

    X_Train_combine['X_Train_combine_'+str(i_model)],\
    X_Test_combine['X_Test_combine_'+str(i_model)] = list_to_data(lists['list_' + str(i_model)],
                                                                      X_train_combined, X_test_combined)
    #*******************************************************************************************************************
    index_a, index_b = section_true(section_list, Y_train_combined, list_gateway_mearge)
    #*******************************************************************************************************************
    for section in range(numebers_section+1):
        index_Y = Y_train_combined[:, 1]
        Max_getway = np.max(np.max(index_Y))
        min_getway = np.min(np.min(index_Y))
        step = (Max_getway - min_getway) / numebers_section
        index['model_' + str(i_model)] = index_a | index_b

    a = index['model_' + str(i_model)]
    X_Train_combine['X_Train_combine_' + str(i_model)] = X_Train_combine['X_Train_combine_' + str(i_model)][a, :]
    Y_Train_combine['Y_Train_combine_' + str(i_model)] = Y_train_combined[a, :]

for i_pre in [0, 1, 2]:
    for i_model in range(numebers_section+1):
        X_train_combined_p, t = preproces(X_train_combined_all, i_pre)
        X_Train_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Train_combine['X_Train_combine_' + str(i_model)], i_pre)
        X_test_combined_p, t = preproces(X_test_combined_all, i_pre)
        X_Test_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Test_combine['X_Test_combine_'+str(i_model)], i_pre)

        if i_model == 0:
            Models[f'regressor_{i_model}'].fit(X_train_combined_p, Y_train_combined)
            Preds[f'Pred_0'] = Models[f'regressor_0'].predict(X_test_combined_p)
            labe_are = Label_area(Preds[f'Pred_0'], numebers_section, Y_train_combined)
            a = evaluation(Y_test_combined, Preds[f'Pred_0'], random_seed_number, i_pre)
        else:
            Models[f'regressor_{i_model}'].fit(X_Train_combine_p['X_Train_combine_' + str(i_model)],
                                               Y_Train_combine['Y_Train_combine_' + str(i_model)])
            Preds[f'Pred_{i_model}'] = Models[f'regressor_{i_model}'].predict(
                X_Test_combine_p['X_Train_combine_' + str(i_model)])

    for i_model in range(1, numebers_section + 1):
        for i_number_label in range(len(X_test_combined)):
            if labe_are[f"model_{i_model - 1}"][i_number_label] == 1:
                Preds['Pred_0'][i_number_label, :] = Preds[f'Pred_{i_model}'][i_number_label, :]

    b = evaluation(Y_test_combined, Preds[f'Pred_0'], random_seed_number, i_pre)
    print(b, a, "\n")
