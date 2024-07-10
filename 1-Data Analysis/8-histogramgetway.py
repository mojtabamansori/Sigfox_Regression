import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('../Dataset/Orginal.csv')

for col in data.columns[:137]:
    plt.figure()
    filtered_data = data[col][data[col] != -200]

    plt.hist(filtered_data)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.xlim([-75, -150])
    plt.title(f'Histogram for column: {col}')
    plt.grid(True)
    plt.savefig(f'Histogram RSSI/histogram_{col}.png')
    plt.close()

print("Histograms generated and saved!")
