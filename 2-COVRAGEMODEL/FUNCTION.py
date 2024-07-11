import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import math
from scipy.spatial import distance

from tqdm import tqdm


def load_data():
    dataset = np.array(pd.read_csv('../Dataset/Orginal.csv'))
    x = dataset[:, :137]
    y = dataset[:, 138:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test

def return_grid(rssi_getway, lat_long):
    x = rssi_getway
    y = lat_long
    def calculate_radius(rssi):
        return 0.00112 * rssi + 0.224

    def create_grid_map(rssi_values, coordinates, point_spacing=0.01):
        new_grid_map = []
        existing_points_set = set(map(tuple, coordinates))
        for idx, rssi in tqdm(enumerate(rssi_values)):
            lat, long = coordinates[idx]
            if rssi > -200:
                radius = calculate_radius(rssi)
                for i in np.arange(-radius, radius, point_spacing):
                    for j in np.arange(-radius, radius, point_spacing):
                        new_lat = lat + i
                        new_long = long + j
                        if (i ** 2 + j ** 2) <= radius ** 2:
                            new_point = (new_lat, new_long)
                            if new_point not in existing_points_set:
                                keep = True
                                for other_point in new_grid_map:
                                    if distance.euclidean(new_point, other_point) < point_spacing:
                                        keep = False
                                        break
                                if keep:
                                    new_grid_map.append(new_point)
                                    existing_points_set.add(new_point)

        return new_grid_map

    new_grid_map = create_grid_map(x, y)
    return new_grid_map