import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('3.xlsx')

# Define x and y values
x_values = [2, 3, 4, 5, 10, 15]
y1_values = df.iloc[::10, 6]  # 7th column, every 10th row
y2_values = df.iloc[::10, 7]  # 8th column, every 10th row

# Create a new figure
fig, ax1 = plt.subplots()

# Plot y1 values
ax1.plot(x_values, y1_values, color='blue')
ax1.set_xlabel('X values')
ax1.set_ylabel('mean_org', color='blue')  # Set y1 label as 'mean_org'
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(x_values, y2_values, color='red')
ax2.set_ylabel('mean_proposed', color='red')  # Set y2 label as 'mean_proposed'
ax2.tick_params(axis='y', labelcolor='red')

# Show the plot
plt.show()
