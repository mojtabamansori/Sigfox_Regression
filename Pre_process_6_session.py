import numpy as np
import pandas as pd

df = pd.read_csv("sigfox_dataset_rural (1).csv")
data_array = df.to_numpy()
X_Train = data_array[:, :137]  # Select columns up to index 136
Y_Train = data_array[:, 138:]  # Select column at index 138
k = [3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4]

for i in k:
    indices = np.argwhere((i <= Y_Train[:,1]) & (Y_Train[:,1] <= i + 0.1))

    filtered_X_Train = X_Train[indices,:]
    filtered_Y_Train = Y_Train[indices,:]

    filtered_X_Train = np.reshape(filtered_X_Train,(filtered_X_Train.shape[0],filtered_X_Train.shape[2]))
    filtered_Y_Train = np.reshape(filtered_Y_Train,(filtered_Y_Train.shape[0],filtered_Y_Train.shape[2]))

    filtered_data = np.concatenate((filtered_X_Train, filtered_Y_Train), axis=1)
    filtered_df = pd.DataFrame(filtered_data)

    # Save the filtered data to an Excel file
    file_name = f"data_{i}_to_{i + 0.1}.xlsx"
    filtered_df.to_csv(file_name, index=False)
