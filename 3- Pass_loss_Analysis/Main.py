import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# بارگذاری فایل CSV
original_file = np.array(pd.read_csv('../Dataset/Orginal.csv'))

# برش داده‌ها
x = original_file[:, :-3]
y = original_file[:, -2:]
num_iterations = min(137, x.shape[1])  # اطمینان از عدم عبور از ابعاد داده

for i2 in range(num_iterations):
    print(f"Processing index: {i2}")

    x_1 = x[:, i2]
    y_1 = y[:, 0]
    y_2 = y[:, 1]

    # فیلتر کردن داده‌ها
    mask = x_1 != -200
    x_1_filtered = x_1[mask]
    y_1_filtered = y_1[mask]
    y_2_filtered = y_2[mask]

    P = 0
    f = 868  # MHz
    P_loss = P - x_1_filtered
    list_of_d = 10 ** ((P_loss - 20 * np.log10(f) - 32.45) / 20)

    # محاسبه قطر
    len_x = np.max(y_1) - np.min(y_1)
    len_y = np.max(y_2) - np.min(y_2)
    diameter = math.sqrt(len_x ** 2 + len_y ** 2)
    diameter = 0.71
    list_new_d = (diameter * list_of_d) / 1068

    # تنظیم مجدد متغیرها برای رسم دایره‌ها
    y_2_plot = np.array(y_1_filtered)  # مرکزهای دایره در محور x
    y_1_plot = np.array(y_2_filtered)  # مرکزهای دایره در محور y
    list_new_d = np.array(list_new_d)  # شعاع‌های دایره‌ها

    # انتخاب ۵ دایره با بزرگترین شعاع‌ها
    top_5_indices = np.argsort(list_new_d)[:]  # ایندکس‌های ۵ شعاع بزرگتر
    top_5_sorted_indices = top_5_indices[np.argsort(list_new_d[top_5_indices])[::-1]]  # مرتب‌سازی نزولی

    # فیلتر کردن مراکز و شعاع‌های مربوطه
    y_1_top_5 = y_1_plot[top_5_sorted_indices]
    y_2_top_5 = y_2_plot[top_5_sorted_indices]
    list_new_d_top_5 = list_new_d[top_5_sorted_indices]

    # ایجاد یک scatter plot با دایره‌های انتخاب‌شده
    fig, ax = plt.subplots(figsize=(24, 16))  # افزایش اندازه شکل به ۱۲x۸ اینچ

    # رسم دایره‌های بزرگتر
    for i in range(len(list_new_d_top_5)):
        circle = plt.Circle((y_1_top_5[i], y_2_top_5[i]), list_new_d_top_5[i],
                            color='b', fill=False, alpha=0.3)  # تنظیم شفافیت
        ax.add_patch(circle)

    # تنظیمات محور‌ها
    ax.set_aspect('equal', 'box')  # برای نسبت مساوی بین محور x و y
    plt.xlim(3.6, 4.4)
    plt.ylim(50.5, 51.6)
    plt.grid(True)

    # عنوان و برچسب‌ها
    plt.title('Scatter plot of top 5 circles with largest radii')
    plt.xlabel('y_1 (X-axis)')
    plt.ylabel('y_2 (Y-axis)')

    # ذخیره نمودار با وضوح بالاتر
    plt.savefig(f'figure/scatter_index_{i2}.png', dpi=300, bbox_inches='tight')  # تنظیم dpi به 300
    plt.close(fig)  # بستن شکل برای صرفه‌جویی در حافظه
