import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
for i in range(40,50):
    dataset = np.array(pd.read_csv(f'parakandi plot/data_x_{i}.csv'))

    # Extract columns
    row = dataset[2:-25, 0]
    org = dataset[2:-25, 1]
    pro = dataset[2:-25, 2]
    P1 = dataset[2:-25, 7]
    P2 = dataset[2:-25, 8]
    len_get1 = dataset[2:-25, 11]
    len_get2 = dataset[2:-25, 12]

    # Function to convert list-like strings to actual lists
    def convert_to_list(list_str):
        # Remove leading and trailing spaces
        list_str = list_str.strip()
        # Remove square brackets and other non-numeric characters
        list_str = list_str.replace('[', '').replace(']', '').replace(',', '').replace('  ', ' ')
        # Replace multiple spaces with a single space
        list_str = ' '.join(list_str.split())
        # Split the string by spaces and convert to integers
        return list(map(int, list_str.split()))

    # Extract and convert lists in get1 and get2
    get1_lists = dataset[2:-25, 9]
    get2_lists = dataset[2:-25, 10]
    get1_combined = np.concatenate([convert_to_list(g) for g in get1_lists])  # Convert strings to lists and concatenate
    get2_combined = np.concatenate([convert_to_list(g) for g in get2_lists])  # Convert strings to lists and concatenate
    plt.figure()
    # Create subplots
    plt.subplot(221)
    plt.title('mean Error')
    plt.plot(row, org, label='Org')
    plt.plot(row, pro, label='Pro')
    plt.ylim([145,220])
    plt.legend()

    plt.subplot(222)
    plt.title('Parakandgi')
    plt.plot(row, P1 - P2, label='lat')
    # plt.plot(row, P2, label='lan')
    # plt.ylim([0,0.25])
    plt.legend()

    plt.subplot(223)
    plt.title('number getways')
    plt.plot(row, len_get1 - len_get2, label='getways section 1')
    # plt.plot(row, len_get2, label='getways section2')
    # plt.ylim([0,138])
    plt.legend()

    # Create side-by-side histograms in subplot 224
    plt.subplot(224)
    plt.title('bar chart getway')
    bins = 137
    plt.hist([get1_combined, get2_combined], bins=bins, label=['Getwayd section 1', 'Getwayd section 2'], histtype='bar', alpha=0.5)
    plt.legend()

    # Show the plot
    print(i)
    plt.savefig(f'parakandi plot/plot_rd_{i}')
