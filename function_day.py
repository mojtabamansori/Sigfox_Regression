import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime


def extract_date(input_string):
    try:
        dt_object = datetime.strptime(input_string, "'%Y-%m-%dT%H:%M:%S%z'")
        date_part = dt_object.date()
        return date_part
    except ValueError:
        print("Input string format is incorrect.")
        return None

def date_day(df):


    df = np.array(df)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    index_day = ['2018-01-09', '2018-01-10', '2018-01-11']

    for number_run, i in enumerate(range(len(df))):
        output_date = extract_date(df[:, 137][i])

        if ((index_day[0] == output_date) or (index_day[1] == output_date) or (index_day[2] == output_date)):
            if X_train == None:
                X_train = df[i, :137]
                Y_train = df[i, 138:]

            else:
                X_train = np.concatenate((X_train, df[i, :137]), axis=0)
                Y_train = np.concatenate((Y_train, df[i, 138:]), axis=0)

        else:
            if X_train == None:
                X_test = df[i, :137]
                Y_test = df[i, 138:]

            else:
                X_test = np.concatenate((X_test, df[i, :137]), axis=0)
                Y_test = np.concatenate((Y_test, df[i, 138:]), axis=0)

    return X_train, X_test, Y_train, Y_test
