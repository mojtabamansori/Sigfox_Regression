import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from vincenty import vincenty
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('sigfox_dataset_rural (1).csv')
X = dataset.iloc[:, :137]
y = dataset.iloc[:, 138:]
X = np.array(X)
y = np.array(y)

'-----------------------------------------------------------------------'
"این بخش 3 جور پری پروسسی بود که در مقاله دوم زدن"
X_norm = (X - np.min(X))/np.min(X)*-1
# X_exp = np.exp((X - np.min(X))/24)/np.exp(np.min(X)*-1/24)
# X_pow = X_norm ** np.e
#
X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42)
# X_train_exp, X_test_exp, _, _ = train_test_split(X_exp, y, test_size=0.30, random_state=42)
# X_train_pow, X_test_pow, _, _ = train_test_split(X_pow, y, test_size=0.30, random_state=42)
# X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
'-------------------------------------------------------------------------'

# poly = PolynomialFeatures(degree=2)
regressor = RandomForestRegressor()

regressor.fit(X_train_norm, y_train)

pred = regressor.predict(X_test_norm)

errors = []
for i in range(len(pred)):
    centroids = pred[i]
    error = vincenty(centroids, y_test[i])
    errors.append(error)


print(f"Mean Error: {np.mean(errors)*1000} meters")
print(f"Median Error: {np.median(errors)*1000} meters")
print(f"R2 Score: {r2_score(y_test, pred)}")
