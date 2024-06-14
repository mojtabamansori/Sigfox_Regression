import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read dataset
dataset = pd.read_csv('../dataset/Orginal.csv').values  # Assuming the path is correct

# Split dataset into features (X) and labels (Y)
X, Y = dataset[:, :137], dataset[:, 138:]

# Iterate over different numbers of sections
for n_s in range(2, 20):
    section_sizes = []

    # Calculate index ranges for each section
    index_Y = Y[:, 1]
    max_value = np.max(index_Y)
    min_value = np.min(index_Y)
    step = (max_value - min_value) / n_s

    for section in range(n_s):
        lower_bound = min_value + (step * section)
        upper_bound = min_value + (step * (section + 1))
        index = (index_Y > lower_bound) & (index_Y <= upper_bound)

        # Filter X and Y based on the index
        X_current = X[index, :]
        Y_current = Y[index, :]
        section_sizes.append(len(X_current))
    plt.figure()
    # Plot bar chart
    plt.bar(range(n_s), section_sizes, color='blue', edgecolor='black')

    # Add titles and labels
    plt.title(f'Bar Chart of Section Sizes (n_s={n_s})')
    plt.xlabel('Section')
    plt.ylabel('Size')

    # Show the plot
    plt.savefig(f'Histogram Count Data/{n_s}')
    print(section_sizes)
