import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure the directory for saving plots exists
output_dir = 'Combined Plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read dataset
dataset = pd.read_csv('../Dataset/Orginal.csv')  # Assuming the path is correct

# Split dataset into features (X) and labels (Y)
X, Y = dataset.iloc[:, :137].values, dataset.iloc[:, 138:].values

# Iterate over different numbers of sections
for n_s in range(2, 20):
    colors = plt.cm.jet(np.linspace(0, 1, n_s))  # Generate colors for each section
    index_Y = Y[:, 1]
    max_value = np.max(index_Y)
    min_value = np.min(index_Y)
    step = (max_value - min_value) / n_s

    section_sizes = []

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    for section in range(n_s):
        lower_bound = min_value + (step * section)
        upper_bound = min_value + (step * (section + 1))

        # Filter indices for each section
        if section == 0:
            indices = (index_Y >= lower_bound) & (index_Y <= upper_bound)
        else:
            indices = (index_Y > lower_bound) & (index_Y <= upper_bound)

        # Scatter Plot
        axs[1].scatter(x=Y[indices, 1], y=Y[indices, 0], color=colors[section])

        # Bar Chart Data
        section_sizes.append(np.sum(indices))

    # Configure Scatter Plot
    axs[1].set_title(f'Scatter Plot of Sections (n_s={n_s})')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')

    # Configure Bar Chart
    axs[0].bar(range(n_s), section_sizes, color='blue', edgecolor='black')
    axs[0].set_title(f'Bar Chart of Section Sizes (n_s={n_s})')
    axs[0].set_xlabel('Section')
    axs[0].set_ylabel('Size')

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{n_s}.png')
    plt.close()
    print(f'n_s={n_s} plot saved.')

print("All combined plots saved successfully.")
