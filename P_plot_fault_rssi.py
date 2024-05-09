import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()
X_Train = data_array[:, :137]
Y_Train = data_array[:, 138:]

for i in range(0, 137):
    print(i)
    color = X_Train[:, i]
    plt.scatter(x=df['Longitude'], y=df['Latitude'], c=color, linewidths=0.01)
    plt.ylim([50.99, 51.21])
    plt.xlim([3.68, 4.3])
    plt.savefig(f'day_plots/-200/figure_{i}.png')
    plt.close()
# mother = 3 5 23 25 27 67
# list_1 = 9 10 11 12 17 19 20 22 26 30 58 61 66 70 71 72 75 82:86 88:92 94 96 97 99 100 101 103 104 105 107 110 118 119
# list_2 = 28 24 18 62 102 126
# list_3 = 0 1 2 4 6 7 8 13 14 15 16 21 29 31 32 33 36 37 38 39 40 43 44 59 60 64 68 73 109
# fault = 18 34 35 41 42 45 46:57 63 65 69 74 76 77 78 79 80 81 87 93 95 98 106 108 111 112 113 114 115 116 117 120:125 127:136