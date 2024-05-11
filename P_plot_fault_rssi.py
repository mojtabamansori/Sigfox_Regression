import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from vincenty import vincenty
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()
X_Train = data_array[:, :137]
Y_Train = data_array[:, 138:]

X_Train, X_test_combined, Y_Train, Y_test_combined = train_test_split(X_Train, Y_Train, test_size=0.3, random_state=42)

# for i in range(0, 137):
#     print(i)
#     color = X_Train[:, i]
#     plt.scatter(x=df['Longitude'], y=df['Latitude'], c=color, linewidths=0.01)
#     plt.ylim([50.99, 51.21])
#     plt.xlim([3.68, 4.3])
#     plt.savefig(f'day_plots/-200/figure_{i}.png')
#     plt.close()

# mother = 3 5 23 25 27 67
# fault = 18 34 35 41 42 45 46:57 63 65 69 74 76 77 78 79 80 81 87 93 95 98 106 108 111 112 113 114 115 116 117 120:125 127:136

list_1 = [9, 10, 11, 12, 17, 19, 20, 22, 26, 30, 58, 61, 66, 70, 71, 72, 75, 82, 83, 84, 85,86, 88, 89, 90, 91 ,
          92, 94, 96, 97, 99, 100, 101, 103, 104, 105, 107, 110, 118, 119, 3, 5, 23, 25, 27, 67]
list_2 = [28, 24, 18, 62, 102, 126, 3, 5, 23, 25, 27, 67]
list_3 = [0, 1, 2, 4, 6, 7, 8, 13, 14, 15, 16, 21, 29, 31, 32, 33, 36, 37, 38, 39, 40, 43, 44, 59, 60, 64, 68, 73, 109, 3, 5, 23, 25, 27, 67]


X_Train_1 = None
X_Train_2 = None
X_Train_3 = None
X_test_1 = None
X_test_2 = None
X_test_3 = None

index_1 = (Y_Train[:, 1] < 3.9)
index_2 = ((3.9 < Y_Train[:, 1]) & (Y_Train[:, 1] < 4.1))
index_3 = (4.1 < Y_Train[:, 1])

X_Train1 = X_Train
X_Train2 = X_Train
X_Train3 = X_Train

Y_Train_1 = Y_Train
Y_Train_2 = Y_Train
Y_Train_3 = Y_Train

for i,number_col in enumerate(list_1):
    if i == 0:
        X_Train_1 = X_Train1[:, number_col].reshape(-1, 1)
        X_test_1 = X_test_combined[:, number_col].reshape(-1, 1)
    else:
        X_Train_1 = np.concatenate((X_Train_1, X_Train1[:, number_col].reshape(-1, 1)), axis=1)
        X_test_1 = np.concatenate((X_test_1, X_test_combined[:, number_col].reshape(-1, 1)), axis=1)

for i,number_col in enumerate(list_2):
    if i == 0:
        X_Train_2 = X_Train2[:, number_col].reshape(-1, 1)
        X_test_2 = X_test_combined[:, number_col].reshape(-1, 1)
    else:
        X_Train_2 = np.concatenate((X_Train_2, X_Train2[:, number_col].reshape(-1, 1)), axis=1)
        X_test_2 = np.concatenate((X_test_2, X_test_combined[:, number_col].reshape(-1, 1)), axis=1)

for i,number_col in enumerate(list_3):
    if i == 0:
        X_Train_3 = X_Train3[:, number_col].reshape(-1, 1)
        X_test_3 = X_test_combined[:, number_col].reshape(-1, 1)
    else:
        X_Train_3 = np.concatenate((X_Train_3, X_Train3[:, number_col].reshape(-1, 1)), axis=1)
        X_test_3 = np.concatenate((X_test_3, X_test_combined[:, number_col].reshape(-1, 1)), axis=1)


print(f'model 1 is running ... x_train = {X_Train_1.shape} y_train = {Y_Train_1.shape}')
model1 = RandomForestRegressor()
model1.fit(X_Train_1, Y_Train_1)

print(f'model 2 is running ... x_train = {X_Train_2.shape} y_train = {Y_Train_2.shape}')
model2 = RandomForestRegressor()
model2.fit(X_Train_2, Y_Train_2)

print(f'model 3 is running ... x_train = {X_Train_3.shape} y_train = {Y_Train_3.shape}')
model3 = RandomForestRegressor()
model3.fit(X_Train_3, Y_Train_3)

errors_eval = []

sum_1 = 0
sum_2 = 0
sum_3 = 0

k = [0, 0, 0]
for i_row in range(len(X_test_1)):
    print(i_row/len(X_test_1))
    i_sum = 0
    x_test_cur = X_test_1[i_row, :]
    pred1 = model1.predict(x_test_cur.reshape(1, -1))
    if (pred1[0][1]) < 3.9:
        k[0] = 0
    else:
        k[0] = 1
        i_sum = i_sum + 1

    x_test_cur = X_test_2[i_row, :]
    pred2 = model2.predict(x_test_cur.reshape(1, -1))
    if ((pred2[0][1]) > 3.9) & ((pred2[0][1]) < 4.1):
        k[1] = 0
    else:
        k[1] = 1
        i_sum = i_sum + 1

    x_test_cur = X_test_3[i_row, :]
    pred3 = model3.predict(x_test_cur.reshape(1, -1))

    if ((pred3[0][1]) > 4.1):
        k[2] = 0
    else:
        k[2] = 1
        i_sum = i_sum + 1

    print(k)
    if ((k[0] == 1) & (k[1] == 0) & (k[2] == 0)):
        errors_eval.append(vincenty(pred1[0], Y_test_combined[i_row, :]))
    if ((k[0] == 0) & (k[1] == 1) & (k[2] == 0)):
        errors_eval.append(vincenty(pred2[0], Y_test_combined[i_row, :]))
    if ((k[0] == 0) & (k[1] == 0) & (k[2] == 1)):
        errors_eval.append(vincenty(pred3[0], Y_test_combined[i_row, :]))

    if ((k[0] == 1) & (k[1] == 0) & (k[2] == 1)):
        pred_1_3 = (pred1[0]+pred3[0])/2
        errors_eval.append(vincenty(pred_1_3, Y_test_combined[i_row, :]))
    if ((k[0] == 1) & (k[1] == 1) & (k[2] == 0)):
        pred_1_2 = (pred1[0] + pred2[0]) / 2
        errors_eval.append(vincenty(pred_1_2, Y_test_combined[i_row, :]))
    if ((k[0] == 0) & (k[1] == 1) & (k[2] == 1)):
        pred_2_3 = (pred2[0] + pred3[0]) / 2
        errors_eval.append(vincenty(pred_2_3, Y_test_combined[i_row, :]))

    if ((k[0] == 1) & (k[1] == 1) & (k[2] == 1)):
        pred_1_2_3 = (pred1[0] + pred2[0]+ pred3[0]) / 3
        errors_eval.append(vincenty(pred_1_2_3, Y_test_combined[i_row, :]))

mean_error = np.mean(errors_eval) * 1000
median_error = np.median(errors_eval) * 1000

print(f"Mean Error: {mean_error} meters")
print(f"Median Error: {median_error} meters")




#     k = np.array([sum_1, sum_2, sum_3])
#     k_i = np.argmax(k)
#     if k_i == 0:
#         pred1 = model1.predict(X_test_1[i_row, :].reshape(1,-1))
#         errors_eval.append(vincenty(pred1[0], Y_test_combined[i_row,:]))
#
#     if k_i == 1:
#         pred2 = model2.predict(X_test_2[i_row, :].reshape(1,-1))
#         errors_eval.append(vincenty(pred2[0], Y_test_combined[i_row, :]))
#
#     if k_i == 2:
#         pred3 = model3.predict(X_test_3[i_row, :].reshape(1,-1))
#         errors_eval.append(vincenty(pred3[0], Y_test_combined[i_row, :]))
#
#     mean_error = np.mean(errors_eval) * 1000
#     median_error = np.median(errors_eval) * 1000
#
# print(f"Mean Error: {mean_error} meters")
# print(f"Median Error: {median_error} meters")

