import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    dataset = np.array(pd.read_csv('../Dataset/Orginal.csv'))
    x = dataset[:, :137]
    y = dataset[:, 138:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test
