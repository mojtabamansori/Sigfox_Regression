import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize lists to store the accumulated data
rows_accumulated = []
org_accumulated = []
pro_accumulated = []
P1_accumulated = []
P2_accumulated = []
len_get1_accumulated = []
len_get2_accumulated = []
get1_combined_all = []
get2_combined_all = []


# Function to convert list-like strings to actual lists
def convert_to_list(list_str):
    list_str = list_str.strip()
    list_str = list_str.replace('[', '').replace(']', '').replace(',', '').replace('  ', ' ')
    list_str = ' '.join(list_str.split())
    return list(map(int, list_str.split()))


# Load datasets and accumulate data
for i in range(40, 50):
    dataset = np.array(pd.read_csv(f'parakandi plot/data_x_{i}.csv'))
    endx = -50
    row = dataset[2:endx, 0]
    org = dataset[2:endx, 1]
    pro = dataset[2:endx, 2]
    P1 = dataset[2:endx, 7]
    P2 = dataset[2:endx, 8]
    len_get1 = dataset[2:endx, 11]
    len_get2 = dataset[2:endx, 12]
    get1_lists = dataset[2:endx, 9]
    get2_lists = dataset[2:endx, 10]

    get1_combined = np.concatenate([convert_to_list(g) for g in get1_lists])
    get2_combined = np.concatenate([convert_to_list(g) for g in get2_lists])

    rows_accumulated.append(row)
    org_accumulated.append(org)
    pro_accumulated.append(pro)
    P1_accumulated.append(P1)
    P2_accumulated.append(P2)
    len_get1_accumulated.append(len_get1)
    len_get2_accumulated.append(len_get2)
    get1_combined_all.append(get1_combined)
    get2_combined_all.append(get2_combined)

# Calculate means
mean_row = np.mean(rows_accumulated, axis=0)
mean_org = np.mean(org_accumulated, axis=0)
mean_pro = np.mean(pro_accumulated, axis=0)
mean_P1 = np.mean(P1_accumulated, axis=0)
mean_P2 = np.mean(P2_accumulated, axis=0)
mean_len_get1 = np.mean(len_get1_accumulated, axis=0)
mean_len_get2 = np.mean(len_get2_accumulated, axis=0)
combined_get1 = np.concatenate(get1_combined_all)
combined_get2 = np.concatenate(get2_combined_all)

# Normalize the data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

plot2 = normalize(mean_P1 - mean_P2)
plot3 = normalize(mean_len_get1 - mean_len_get2)

# Compute the discrete derivative
def discrete_derivative(data):
    return np.diff(data)

derivative_plot2 = discrete_derivative(plot2)
derivative_plot3 = discrete_derivative(plot3)

# Function to add straight lines
# def add_straight_lines(ax):
#     ax.axvline(x=10, color='red', linestyle='--')
#     ax.axvline(x=16, color='red', linestyle='--')
#     ax.axvline(x=25, color='blue', linestyle='--')
#     ax.axvline(x=27, color='blue', linestyle='--')


# Plotting the results
plt.figure()

plt.subplot(321)
plt.title('moshtage(Parakandgi_section1 - Parakandgi_section1)')
plt.plot(mean_row[:-1], derivative_plot2, label='Normalized Derivative of lat')
plt.legend()
# add_straight_lines(plt.gca())

plt.subplot(322)
plt.title('Parakandgi_section1 - Parakandgi_section1')
plt.plot(mean_row, plot2, label='Normalized lat')
# add_straight_lines(plt.gca())
plt.legend()

plt.subplot(323)
plt.plot(mean_row[:-1], derivative_plot3, label='Normalized Derivative of Gateways')
plt.title('moshtage(Number of Gateways section 1 - Number of Gateways section 2)')

plt.legend()
# add_straight_lines(plt.gca())

plt.subplot(324)
plt.title('Number of Gateways section 1 - Number of Gateways section 2')
plt.plot(mean_row, plot3, label='Normalized Gateways section 1')
# add_straight_lines(plt.gca())
plt.legend()

plt.subplot(325)
plt.title('conv3node(mean(1,3))')
plt.plot(mean_row[:-1], np.convolve(1-(derivative_plot2+(derivative_plot3))/2, np.array([1, 1, 1]), mode='same'), label='Normalized Derivative of Gateways')
plt.legend()
# add_straight_lines(plt.gca())

plt.subplot(326)
plt.title('mse')
plt.plot(row,mean_org)
plt.plot(row,mean_pro)

# plt.savefig('parakandi_plot/mean_plot')
plt.show()
