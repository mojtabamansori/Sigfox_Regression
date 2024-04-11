import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame
df = pd.read_csv("sigfox_dataset_rural (1).csv")

# Define the first linear equation
def linear_eq1(x):
    return 0.4 * (x - 3.65) + 51

# Define the second linear equation
def linear_eq2(x):
    return 0.20 * (x - 4) + 51.05

# Create masks for points inside and outside the shaded region
mask_in_shaded_region = (df['Longitude'] >= min(df['Longitude'])) & (df['Longitude'] <= max(df['Longitude'])) & \
                        (df['Latitude'] >= np.minimum(linear_eq1(df['Longitude']), linear_eq2(df['Longitude']))) & \
                        (df['Latitude'] <= np.maximum(linear_eq1(df['Longitude']), linear_eq2(df['Longitude'])))
mask_outside_shaded_region = ~mask_in_shaded_region

# Split the data into train and test sets based on the masks
X_train, X_test, y_train, y_test = df[mask_in_shaded_region][['Longitude', 'Latitude']], \
                                    df[mask_outside_shaded_region][['Longitude', 'Latitude']], \
                                    df[mask_in_shaded_region][['Longitude', 'Latitude']], \
                                    df[mask_outside_shaded_region][['Longitude', 'Latitude']]

# Plot Latitude and Longitude with color based on the order of RX Time
plt.scatter(x=df['Longitude'], y=df['Latitude'], c='gray', label='All data')
plt.scatter(x=X_train['Longitude'], y=X_train['Latitude'], c='blue', label='Train data')
plt.scatter(x=X_test['Longitude'], y=X_test['Latitude'], c='red', label='Test data')

# Add colorbar
plt.colorbar(label='Order of RX Time')

# Add labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Plot the first line
x_values1 = np.linspace(min(df['Longitude']) - 10, max(df['Longitude']) + 10, 100)
plt.plot(x_values1, linear_eq1(x_values1), color='red', linestyle='--', label='y = 0.4x + 51')

# Plot the second line
x_values2 = np.linspace(min(df['Longitude']) - 10, max(df['Longitude']) + 10, 100)
plt.plot(x_values2, linear_eq2(x_values2), color='blue', linestyle='--', label='y = 0.20x + 51.05')

# Add a legend
plt.legend

plt.savefig('main_way.pdf')


# Show the plot
plt.show()

# Save train and test points as CSV files individually
X_train_with_all_columns = df[mask_in_shaded_region]
X_test_with_all_columns = df[mask_outside_shaded_region]

# Save train and test points with all columns as CSV files individually
X_train_with_all_columns.to_csv('train_points_with_all_columns.csv', index=False)
X_test_with_all_columns.to_csv('test_points_with_all_columns.csv', index=False)