from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from vincenty import vincenty


def load_data():
    dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
    X, Y = dataset[:, :137], dataset[:, 138:]
    X_train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, Y_Train, X_Test, Y_Test