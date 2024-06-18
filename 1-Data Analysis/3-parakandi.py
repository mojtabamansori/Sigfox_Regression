from function import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


for rd in range(40,50):
    result = {"orginal":[], "proposed":[], "start_section":[], "Final_section":[], "Error":[], "number hat":[], "P1":[], "P2":[], "get1":[], "get2":[], "len_get1":[], "len_get2":[]}
    i333 = 0
    i333 = i333 + 1
    label_section_value = {"section_number": [], "start_section": [], "final_section": []}
    numbers_section = 2
    re1 = 2
    last_numbers_section = 2

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

    X_train_combined, Y_train_combined, X_test_combined,\
        Y_test_combined = load_date_def(rd, (numbers_section + 1))

    section_list = return_section_list(numbers_section + 1,
                                       np.max(np.max(Y_train_combined[:, 1])), np.min(np.min(Y_train_combined[:, 1])))


    c_1 = 4.33295822 - 3.6850648

    for i_c in np.arange(0.02, c_1, 0.005):
        print(f'rd = {rd} done =',(i_c/c_1))
        section_list[0, 0] = 3.6850648
        section_list[0, 1] = 3.6850648 + i_c
        section_list[1, 0] = 3.6850648 + i_c
        section_list[1, 1] = 4.33295822
        label_section_value = {"section_number": [], "start_section": [], "final_section": []}
        for i_temp in range(len(section_list)):
            label_section_value["section_number"].append([i_temp])
            label_section_value["start_section"].append([section_list[i_temp, 0]])
            label_section_value["final_section"].append([section_list[i_temp, 1]])

        useful_section_getway = f_e_mean_std(X_train_combined, Y_train_combined, numbers_section,section_list)
        lists = list_getways(useful_section_getway, numbers_section)
        re2 = numbers_section
        re3 = label_section_value["start_section"]
        re4 = label_section_value["final_section"]

        Models, Preds, Preds_section = {}, {}, {}
        y_section, X_Train_combine, X_Train_combine_p = {}, {}, {}
        Y_Train_combine, X_Test_combine, X_Test_combine_p= {}, {}, {}
        index = {}
        Errors = {}
        X_train_combined_all, X_test_combined_all = list_to_data(list_fualt_not, X_train_combined, X_test_combined)
        for i_model in range(numbers_section + 1):
            Models['regressor_' + str(i_model)] = RandomForestRegressor()
            Preds['Pred_' + str(i_model)] = None
            X_Train_combine['X_Train_combine_' + str(i_model)], \
                        X_Test_combine['X_Test_combine_'+str(i_model)] = list_to_data(lists['list_' + str(i_model)],
                                                                                      X_train_combined, X_test_combined)

            index = index_section(numbers_section, Y_train_combined, i_model, index, label_section_value)
            a1 = index['model_' + str(i_model)][0]
            X_Train_combine['X_Train_combine_' + str(i_model)] = X_Train_combine['X_Train_combine_' + str(i_model)][a1, :]
            Y_Train_combine['Y_Train_combine_' + str(i_model)] = Y_train_combined[a1, :]

        if (len(X_Train_combine['X_Train_combine_' + str(i_model)])) != 0:
            for i_model in range(numbers_section + 1):
                X_train_combined_p, t = preproces(X_train_combined_all, 0)

                X_Train_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Train_combine['X_Train_combine_' + str(i_model)], 0)
                X_test_combined_p, t = preproces(X_test_combined_all, 0)
                X_Test_combine_p['X_Train_combine_' + str(i_model)], t = preproces(X_Test_combine['X_Test_combine_'+str(i_model)], 0)

                if i_model == 0:
                    Models[f'regressor_{i_model}'].fit(X_train_combined_p, Y_train_combined)
                    Preds[f'Pred_0'] = Models[f'regressor_0'].predict(X_test_combined_p)
                    labe_are = Label_area_new_way(Preds[f'Pred_0'], numbers_section, label_section_value)
                    a = evaluation(Y_test_combined, Preds[f'Pred_0'], 42, 0)
                else:
                    Models[f'regressor_{i_model}'].fit(X_Train_combine_p['X_Train_combine_' + str(i_model)],
                                                       Y_Train_combine['Y_Train_combine_' + str(i_model)])
                    Preds[f'Pred_{i_model}'] = Models[f'regressor_{i_model}'].predict(
                        X_Test_combine_p['X_Train_combine_' + str(i_model)])
            Err = {}
            for issss in range(1, numbers_section+1):
                Err[f'true_{issss}'] = []
                Err[f'hat_{issss}'] = []

            Err[f'number hat'] = []

            for i_model in range(1, numbers_section + 1):
                for i_number_label in range(len(X_test_combined)):
                    if labe_are[f"model_{i_model - 1}"][0, i_number_label] == 1:
                        Preds['Pred_0'][i_number_label, :] = Preds[f'Pred_{i_model}'][i_number_label, :]
                        Err[f'true_{i_model}'].append(Y_test_combined[i_number_label, :])
                        Err[f'hat_{i_model}'].append(Preds[f'Pred_{i_model}'][i_number_label, :])
                    else:
                        if np.sum(labe_are[f"model_{i_model - 1}"], axis=0)[i_number_label] == 1:
                            Preds['Pred_0'][i_number_label, :] = Preds[f'Pred_{i_model}'][i_number_label, :]
                            Err[f'true_{i_model}'].append(Y_test_combined[i_number_label, :])
                            Err[f'hat_{i_model}'].append(Preds[f'Pred_{i_model}'][i_number_label, :])

                Err[f'number hat'].append(np.sum(labe_are[f"model_{i_model - 1}"]))
                Errors[f'regressor_{i_model}'] = evaluation(Err[f'true_{i_model}'], Err[f'hat_{i_model}'],
                                                            42, 0)

            b = evaluation(Y_test_combined, Preds[f'Pred_0'], 42, 0)
            re5 = a
            re6 = b
            c = Errors
            ac = Err[f'number hat']
            # print(i_c, b, a, "\n")
        else:
            print('fail')

        (result["orginal"]).append(re5)
        (result["proposed"]).append(re6)
        (result["start_section"]).append(re3)
        (result["Final_section"]).append(re4)
        (result["Error"]).append(c)
        (result["number hat"]).append(ac)
        aaa = Y_Train_combine['Y_Train_combine_1']
        bbb = Y_Train_combine['Y_Train_combine_2']
        (result["P1"]).append(np.std(aaa[:, 1]))
        (result["P2"]).append(np.std(bbb[:, 1]))
        (result["get1"]).append(lists['list_1'])
        (result["get2"]).append(lists['list_2'])
        (result["len_get1"]).append(len(lists['list_1']))
        (result["len_get2"]).append(len(lists['list_2']))

        DF = pd.DataFrame(result)
        DF.to_csv(f"parakandi plot/data_x_{rd}.csv")
