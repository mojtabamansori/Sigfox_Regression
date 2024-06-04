import numpy as np
import pandas as pd

file_path = r"../Dataset/Orginal.csv"
df = pd.read_csv(file_path)
df = np.array(df)
X = df[:, :137]
Y = df[:, 138:]

# n_section = 5
#
# min_y = np.min(Y[:, 1])
# max_y = np.max(Y[:, 1])
# step_y = (max_y - min_y) / n_section
#
# list_section = np.zeros((n_section, 2))
# for i in range(n_section):
#     list_section[i,0] = min_y + (i * step_y)
#     list_section[i,1] = min_y + ((i+1) * step_y)
# print(list_section)
# x_train = {}
#
# for i in range(n_section):
#     index = ((list_section[i, 0] < Y[:, 1]) & (list_section[i, 1] > Y[:, 1]))
#     x_train[f'data_{i}'] = X[index, :]
#     print(X[index, :].shape)




