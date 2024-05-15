import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt

number_section = 20

df = pd.read_csv(f'sigfox_dataset_rural (1).csv')
dataset = np.array(df)
X = dataset[:, :137]
Y = dataset[:, 138:]
data_plot_mean = np.zeros((number_section, 137))
data_plot_std = np.zeros((number_section, 137))
for section in range(number_section):
    index_Y = Y[:, 1]
    Max_getway = np.max(np.max(index_Y))
    min_getway = np.min(np.min(index_Y))
    step = (Max_getway - min_getway)/number_section
    data_getway = X[(((min_getway + (step * section)) < index_Y) & ((min_getway + (step * (section+1))) > index_Y))]
    mean_getway = np.mean(data_getway, axis=0)
    std_getway = np.std(np.float32(data_getway), axis=0)
    data_plot_mean[section, :] = mean_getway
    data_plot_std[section, :] = std_getway


getway_useful = []
for number_getway in range(137):
    getway_useful.append(1)
    for number_sections in range(number_section):
        if data_plot_mean[number_sections, number_getway] != -200:
            getway_useful.append(number_sections)
            getway_useful.append(number_getway)

print(getway_useful)


# for number_getway in range(137):
#     plt.figure()
#     plt.plot(data_plot_mean[:, number_getway])
#     plt.plot((data_plot_mean[:, number_getway] + data_plot_std[:, number_getway]), '*')
#     plt.plot((data_plot_mean[:, number_getway] - data_plot_std[:, number_getway]), '*')
#     plt.xlabel('section')
#     plt.ylabel('mean and std')
#     plt.legend(['mean', 'std +', 'std -'])
#     plt.savefig(f'result\mean_std\getway_{number_getway + 1}')
#
