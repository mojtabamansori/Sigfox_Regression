import matplotlib.pyplot as plt
import pandas as pd


# Read the CSV file into a DataFrame
for i in range(1, 81):
    print(i)
    df = pd.read_csv(f'days_csv_results/day_data_{i}.csv')

    # Create a new figure for each plot
    # plt.figure()

    # Create a range of integers representing the order of rows
    order = range(len(df))

    # Plot Latitude and Longitude with color based on the order of RX Time
    plt.scatter(x=df['Longitude'], y=df['Latitude'])

    # Add colorbar


    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.ylim([50.99,51.21])
    plt.xlim([3.68, 4.3])
    plt.savefig(f'day_plots/not_close/figure_{i}.png')

    # plt.close()
