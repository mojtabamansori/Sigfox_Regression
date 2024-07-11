import matplotlib.pyplot as plt
from FUNCTION import *


x_train, x_test, y_train, y_test = load_data()

for i2 in range(4,137):
    point = return_grid(x_train[:, i2], y_train)
    df = pd.DataFrame(point, columns=['lat', 'long'])
    df.to_csv(f'csv\point_data{i2}.csv', index=False)
    plt.figure()
    for i in point:
        plt.scatter(i[1], i[0], alpha=0.4, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RSSI Scatter Plot')
    plt.xlim(3.6, 4.4)
    plt.ylim(50.5, 52)
    plt.savefig(f'plot_grid_point\grid_map{i2}.png')
    plt.close()