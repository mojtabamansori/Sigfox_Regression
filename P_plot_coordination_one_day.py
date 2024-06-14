import matplotlib.pyplot as plt
import pandas as pd

for i in range(1, 3):
    print(i)
    df = pd.read_csv(f'Dataset/Orginal.csv')

    order = range(len(df))
    plt.scatter(x=df['Longitude'], y=df['Latitude'])

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.ylim([50.99,51.21])
    plt.xlim([3.68, 4.3])
    plt.savefig(f'day_plots/figure_{i}.png')