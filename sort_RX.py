import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("sigfox_dataset_rural (1) - Copy.csv")

# Convert 'RX Time' column to datetime
df['RX Time'] = pd.to_datetime(df['RX Time'].str.strip("'"))

# Sort the DataFrame by 'RX Time'
df_sorted = df.sort_values(by='RX Time')

# Print the sorted DataFrame

df_sorted.to_csv('sorted_file.csv', index=False)