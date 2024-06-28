import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your dataset
dataset = np.array(pd.read_csv('../Dataset/Orginal.csv'))
Input = dataset[:, :137]
Output = dataset[:, 138:]

# Create a scatter plot for each RSSI value
for i in range(137):
    plt.figure()
    for i2 in range(len(Input)):
        print(i2)
        rssi = Input[i2, i]
        y = Output[i2, 0]
        x = Output[i2, 1]
        if (rssi+200) != 0:
            plt.scatter(x, y, s=((rssi+200)*2), alpha=0.4)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RSSI Scatter Plot')
    plt.xlim(np.min(Output[:, 1]), np.max(Output[:, 1]))
    plt.ylim(np.min(Output[:, 0]), np.max(Output[:, 0]))
    plt.grid(True)
    plt.savefig(f'plot_RF_COVAERAGE/(rssi+200)zarb2/getway_{i}.png')
