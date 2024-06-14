import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = np.array(pd.read_csv(f'..\dataset\Orginal.csv'))
X, Y = dataset[:, :137], dataset[:, 138:]
