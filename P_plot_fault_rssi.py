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
# mother = 3 5
# list_1 = 9 10 11 12 17 19 20 22
# list_2 =
# list_3 = 0 1 2 4 6 7 8 13 14 15 16 21
# fault = 18