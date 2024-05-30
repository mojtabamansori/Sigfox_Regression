from function import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
for i_sec_to in range(10, 11):
    numebers_section = i_sec_to
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

    dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
    X, Y = dataset[:, :137], dataset[:, 138:]

    list_random_seed = range(42, 43, 1)
    X_train_combined, Y_train_combined, X_test_combined, Y_test_combined = load_date_def(list_random_seed, (i_sec_to+1))
    section_list = return_section_list(numebers_section+1, np.max(np.max(Y_train_combined[:, 1])), np.min(np.min(Y[:, 1])))
    useful_section_getway = f_e_mean_std(X_train_combined, Y_train_combined, numebers_section)
    lists = list_getways(useful_section_getway, numebers_section)

    list_gateway_mearge = []
    for number_mearge in range(numebers_section):
        A_S_B = list_change_section_r1(lists, numebers_section, number_mearge, list_gateway_mearge, section_list)
        if A_S_B == 5:
            pass
        if A_S_B == 0:
            lists, numebers_section, list_gateway_mearge, section_list = list_change_section(lists, numebers_section, number_mearge, list_gateway_mearge, section_list)
    print(i_sec_to,' is number section and number section after mearg is ===>  ', numebers_section)

    Models, Preds, Preds_section = {}, {}, {}
    y_section, X_Train_combine, X_Train_combine_p = {}, {}, {}
    Y_Train_combine, X_Test_combine, X_Test_combine_p= {}, {}, {}
    index = {}
    X_train_combined_all, X_test_combined_all = list_to_data(list_fualt_not,
                                                         X_train_combined, X_test_combined)

    for i_model in range(numebers_section+1):
        Models['regressor_' + str(i_model)] = RandomForestRegressor()
        Preds['Pred_' + str(i_model)] = None

        X_Train_combine['X_Train_combine_'+str(i_model)],\
        X_Test_combine['X_Test_combine_'+str(i_model)] = list_to_data(lists['list_' + str(i_model)],
                                                                          X_train_combined, X_test_combined)

        index = index_section(numebers_section, Y_train_combined, i_model, index, section_list)

        a = index['model_' + str(i_model)]
        X_Train_combine['X_Train_combine_' + str(i_model)] = X_Train_combine['X_Train_combine_' + str(i_model)][a, :]
        Y_Train_combine['Y_Train_combine_' + str(i_model)] = Y_train_combined[a, :]

    if len(X_Train_combine['X_Train_combine_' + str(i_model)]) == 0:
        print(i_sec_to, 'fail')

    if len(X_Train_combine['X_Train_combine_' + str(i_model)]) != 0:
        for i_model in range(numebers_section+1):
            X_train_combined_p, t = preproces(X_train_combined_all, 0)
            X_Train_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Train_combine['X_Train_combine_' + str(i_model)], 0)
            X_test_combined_p, t = preproces(X_test_combined_all, 0)
            X_Test_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Test_combine['X_Test_combine_'+str(i_model)], 0)

            if i_model == 0:
                Models[f'regressor_{i_model}'].fit(X_train_combined_p, Y_train_combined)
                Preds[f'Pred_0'] = Models[f'regressor_0'].predict(X_test_combined_p)
                labe_are = Label_area_new_way(Preds[f'Pred_0'], numebers_section, section_list)
                a = evaluation(Y_test_combined, Preds[f'Pred_0'], 42, 0)
            else:
                Models[f'regressor_{i_model}'].fit(X_Train_combine_p['X_Train_combine_' + str(i_model)],
                                                   Y_Train_combine['Y_Train_combine_' + str(i_model)])
                Preds[f'Pred_{i_model}'] = Models[f'regressor_{i_model}'].predict(
                    X_Test_combine_p['X_Train_combine_' + str(i_model)])

        for i_model in range(1, numebers_section + 1):
            for i_number_label in range(len(X_test_combined)):
                if labe_are[f"model_{i_model - 1}"][i_number_label] == 1:
                    Preds['Pred_0'][i_number_label, :] = Preds[f'Pred_{i_model}'][i_number_label, :]

        b = evaluation(Y_test_combined, Preds[f'Pred_0'], 42, 0)
        print(i_sec_to, b, a, "\n")
















