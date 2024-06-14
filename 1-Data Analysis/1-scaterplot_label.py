import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure the directory for saving plots exists
output_dir = 'Scatter Plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read dataset
dataset = pd.read_csv('../Dataset/Orginal.csv')  # Assuming the path is correct

# Split dataset into features (X) and labels (Y)
Y = dataset.iloc[:, 138:].values
# Iterate over different numbers of sections
for n_s in range(2, 20):
    colors = plt.cm.jet(np.linspace(0, 1, n_s))  # Generate colors for each section
    index_Y = Y[:, 1]
    max_value = np.max(index_Y)
    min_value = np.min(index_Y)
    step = (max_value - min_value) / n_s

    plt.figure()

    for section in range(n_s):
        lower_bound = min_value + (step * section)
        upper_bound = min_value + (step * (section + 1))

        # Filter indices for each section
        if section == 0:
            indices = (index_Y >= lower_bound) & (index_Y <= upper_bound)
        else:
            indices = (index_Y > lower_bound) & (index_Y <= upper_bound)

        # Plotting the data for the current section
        print('plot is running')
        # plt.scatter(x=Y[indices, 0], y=Y[indices, 1], color=colors[section])
        plt.scatter(x=Y[indices, 1], y=Y[indices, 0])

    plt.title(f'Scatter Plot of Sections (n_s={n_s})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.ylim([50.99, 51.21])
    # plt.xlim([3.68, 4.3])
    plt.savefig(f'{output_dir}/{n_s}.png')
    plt.close()
    print(f'n_s={n_s} plot saved.')

print("All plots saved successfully.")
