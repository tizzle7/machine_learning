# feature_scaling.py

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

wine = pd.read_csv("wine.data", header=None)
wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
                "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
                "Hue", "OD280/OD315 of diluted wines", "Proline"]

# use train_test_split function to generaze separate training and test sets,
# assign 70% of the data to the training set and 30% to the test set
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# use the min-max scaling procedure to normalize features
mmsc = MinMaxScaler()
X_train_norm = mmsc.fit_transform(X_train)
X_test_norm = mmsc.transform(X_test)

# standardize the features using the corresponding sci-learn scaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print(X_train_std)

