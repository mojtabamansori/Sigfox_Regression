import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import distance

# داده‌ها
x = np.array([-75, -100, -120, -185])  # آرایه RSSI ها
y = np.array([[3.65, 51.3890], [3.75, 51.3891], [3.85, 51.3892], [3.95, 51.3]])  # آرایه lat و long


# محاسبه شعاع دایره‌ها
def calculate_radius(rssi):
    return 0.00112 * rssi + 0.224


# تابع برای ایجاد گرید مپ جدید
def create_grid_map(rssi_values, coordinates, point_spacing=0.01):
    new_grid_map = []
    existing_points_set = set(map(tuple, coordinates))

    for idx, rssi in enumerate(rssi_values):
        lat, long = coordinates[idx]
        radius = calculate_radius(rssi)

        # ایجاد نقاط درون دایره
        for i in np.arange(-radius, radius, point_spacing):
            for j in np.arange(-radius, radius, point_spacing):
                new_lat = lat + i
                new_long = long + j

                # بررسی اینکه نقطه درون دایره باشد
                if (i ** 2 + j ** 2) <= radius ** 2:
                    new_point = (new_lat, new_long)
                    # اگر نقطه جدید در نقاط گرید مپ قبلی نباشد و فاصله مناسب داشته باشد، به لیست اضافه شود
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


# فراخوانی تابع
new_grid_map = create_grid_map(x, y)

# نمایش نقاط گرید مپ جدید
plt.figure(figsize=(10, 8))

for i in y:
    plt.scatter(i[0], i[1], alpha=1, color='red', label='Old Points')

for point in new_grid_map:
    plt.scatter(point[0], point[1], alpha=0.4, color='blue')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Grid Map of Points Inside Circles')
plt.legend(['Old Points', 'New Grid Points'])
plt.show()
