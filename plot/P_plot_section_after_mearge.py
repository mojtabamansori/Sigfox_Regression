import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from vincenty import vincenty
from scipy.spatial import distance
from sklearn.metrics import r2_score


df = pd.read_csv("../Rebuild_final_code/data1.csv")
data_array = df.to_numpy()

data = data_array[:, 3:5]
first_column_values = data[:, 0]
second_column_values = data[:, 1]

height = 1
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(3, 6, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')

for i_row in range(len(first_column_values)):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)')
    start = ast.literal_eval(first_column_values[i_row])
    for i in start:
        print(i)
        plt.axvline(x=i, linestyle='--', label='Middle Line')
    plt.title(f'i_row')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.savefig(f'E:/Sigfox_Regression/result/meargw_section/{i_row}.png')


