from function import *
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


numebers_section = 3
dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
X, Y = dataset[:, :137], dataset[:, 138:]
useful_section_getway = f_e_mean_std(X, Y, numebers_section)
lists = list_getways(useful_section_getway, numebers_section)




list_1 = lists['list_0']
list_2 = lists['list_1']
list_3 = lists['list_2']

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
for n_1 in range(75, 76, 5):
    for n_2 in range(75, 125, 5):
        for n_3 in range(75, 76, 5):
            for random_seed_number in range(42, 43, 1):
                regressor = RandomForestRegressor()
                regressor_1 = RandomForestRegressor(n_estimators=n_1, criterion='friedman_mse', max_depth=n_3)
                regressor_2 = RandomForestRegressor(n_estimators=n_2, criterion='friedman_mse', max_depth=n_2)
                regressor_3 = RandomForestRegressor(n_estimators=n_3, criterion='friedman_mse', max_depth=n_1)

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

                X_train_combined_all, X_test_combined_all = list_to_data(list_fualt_not,
                                                                         X_train_combined, X_test_combined)
                X_train_combined_1, X_test_combined_1 = list_to_data(list_1, X_train_combined, X_test_combined)
                X_train_combined_2, X_test_combined_2 = list_to_data(list_2, X_train_combined, X_test_combined)
                X_train_combined_3, X_test_combined_3 = list_to_data(list_3, X_train_combined, X_test_combined)

                index_1 = (Y_train_combined[:, 1] < 3.94)
                index_2 = ((3.94 < Y_train_combined[:, 1]) & (Y_train_combined[:, 1] < 4.03))
                index_3 = (4.03 < Y_train_combined[:, 1])


            # **************************************************************

                X_train_combined_1 = X_train_combined_1[index_1, :]
                X_train_combined_2 = X_train_combined_2[index_2, :]
                X_train_combined_3 = X_train_combined_3[index_3, :]

                Y_train_combined_1 = Y_train_combined[index_1, :]
                Y_train_combined_2 = Y_train_combined[index_2, :]
                Y_train_combined_3 = Y_train_combined[index_3, :]

                # Y_train_combined_1 = Y_train_combined
                # Y_train_combined_2 = Y_train_combined
                # Y_train_combined_3 = Y_train_combined

            # **************************************************************


                for i_pre in [0, 1, 2]:
                    X_train_combined_p, t = preproces(X_train_combined_all, i_pre)
                    X_train_combined_p_1, t = preproces(X_train_combined_1, i_pre)
                    X_train_combined_p_2, t = preproces(X_train_combined_2, i_pre)
                    X_train_combined_p_3, t = preproces(X_train_combined_3, i_pre)

                    X_test_combined_p, t = preproces(X_test_combined_all, i_pre)
                    X_test_combined_p_1, t = preproces(X_test_combined_1, i_pre)
                    X_test_combined_p_2, t = preproces(X_test_combined_2, i_pre)
                    X_test_combined_p_3, t = preproces(X_test_combined_3, i_pre)



                    # print('model all')
                    regressor.fit(X_train_combined_p, Y_train_combined)
                    # print('model all done')
                    pred = regressor.predict(X_test_combined_p)

                    # print('model 1')
                    regressor_1.fit(X_train_combined_p_1, Y_train_combined_1)
                    # print('model 1 done')
                    # print('model 2')
                    regressor_2.fit(X_train_combined_p_2, Y_train_combined_2)
                    # print('model 2 done')
                    # print('model 3')
                    regressor_3.fit(X_train_combined_p_3, Y_train_combined_3)
                    # print('model 3 done')

                    pred_1 = regressor_1.predict(X_test_combined_p_1)
                    pred_2 = regressor_2.predict(X_test_combined_p_2)
                    pred_3 = regressor_3.predict(X_test_combined_p_3)

                    print('model************************')
                    a = evaluation(Y_test_combined, pred, random_seed_number, i_pre)

                    labe_are = Label_area(pred)

                    pred_1_2 = pred_1[labe_are == 0]
                    pred_2_2 = pred_2[labe_are == 1]
                    pred_3_2 = pred_3[labe_are == 2]
                    Y_test_combined_1 = Y_test_combined[labe_are == 0]
                    Y_test_combined_2 = Y_test_combined[labe_are == 1]
                    Y_test_combined_3 = Y_test_combined[labe_are == 2]
                    evaluation1(Y_test_combined_1, pred_1_2, random_seed_number, i_pre)
                    evaluation1(Y_test_combined_2, pred_2_2, random_seed_number, i_pre)
                    evaluation1(Y_test_combined_3, pred_3_2, random_seed_number, i_pre)

                    for i_number_label in range(len(X_test_combined)):
                        if labe_are[i_number_label] == 0:
                            pred[i_number_label, :] = pred_1[i_number_label, :]
                        if labe_are[i_number_label] == 1:
                            pred[i_number_label, :] = pred_2[i_number_label, :]
                        if labe_are[i_number_label] == 2:
                            pred[i_number_label, :] = pred_3[i_number_label, :]

                    print('model 3                      model all')
                    b = evaluation(Y_test_combined, pred, random_seed_number, i_pre)
                    print(b, a, "\n")
                    if (b-a) < 0:
                        print(n_1, n_2, n_3)






