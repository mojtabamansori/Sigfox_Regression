import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt

k2 = ['org_35','org_36','org_37','org_38','org_39','org_40','org_41','session_36','session_37','session_38','session_39','session_40','session_41']
# Load your train and test datasets
for hn,algo in enumerate(k2):
    errors = np.array(pd.read_csv(f'error/{algo}_error.csv'))
    k = []
    for ii in errors:
        for i in ii:
            i = float(i.replace('[[','').replace(']]',''))
            k.append(i)

    plt.figure()
    # plt.subplot(1, 5, hn+1)
    xd = 800
    plt.hist(k, range=[0,xd]) # Plot histogram with 20 bins
    plt.xlabel('Error (meters)')
    plt.ylabel('Frequency')
    plt.ylim(0,500)
    plt.xlim(0,xd)
    plt.title(f'{algo} ')
    plt.savefig(f'error/histogram_ylim_{algo}.png')
    print(f"{algo}-error ==> Mean error:", np.mean(k))
    print(f"{algo}-error ==> Median error:", np.median(k))
